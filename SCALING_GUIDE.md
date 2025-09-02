# Scaling Guide: Purchase-Inventory Discrepancy Analyzer

## Overview

This document provides comprehensive guidance for scaling the Purchase-Inventory Discrepancy Analyzer to handle large datasets (1M+ records). The current implementation works well for moderate datasets but faces bottlenecks when processing high-volume data.

## Current Architecture Analysis

### Existing Implementation Strengths
- **Robust Error Handling**: Exponential backoff retry mechanism for API rate limiting
- **Data Preservation**: Smart preservation of user comments and resolution status
- **Comprehensive Reporting**: Multiple report types with rich formatting
- **Modular Design**: Well-structured class-based architecture

### Current Bottlenecks for High-Volume Data

#### 1. Google Sheets API Limitations
```python
# Current approach - sequential API calls
def read_sheet_data(self, sheet_id, sheet_name="Sheet1"):
    all_values = self._api_call_with_retry(lambda: worksheet.get_all_values())
    # Processes entire dataset in memory at once
```

**Issues:**
- API quota limits (100 requests per 100 seconds per user)
- Read/write operations are synchronous
- Memory consumption scales linearly with dataset size
- Network latency compounds with dataset size

#### 2. In-Memory Processing
```python
# Memory-intensive operations
df = pd.DataFrame(data_rows, columns=headers)  # Entire dataset in memory
grouped = purchase_df.groupby(['DATE', 'PURCHASE OFFICER NAME']).agg({...})  # Full dataset aggregation
```

**Issues:**
- RAM requirements grow with dataset size
- No streaming or chunked processing
- Pandas operations on large datasets become slow

#### 3. Sequential Processing Pattern
```python
# Linear processing flow
purchase_grouped = self.process_purchase_data(purchase_df)
inventory_processed = self.process_inventory_data(inventory_df) 
weight_report = self.generate_weight_discrepancy_report(purchase_grouped, inventory_processed)
```

## Scaling Approaches Analysis

### Option 1: AWS Glue + Ray Assessment

#### Ray Compatibility Analysis
**Current Code Compatibility: ⚠️ PARTIAL**

Ray works best with:
- CPU-intensive computations ✅
- Parallelizable data transformations ✅
- Distributed datasets ❌ (Google Sheets API)

**What would need refactoring:**
```python
# Current approach
def process_purchase_data(self, purchase_df):
    grouped = purchase_df.groupby(['DATE', 'PURCHASE OFFICER NAME']).agg({...})
    
# Ray-compatible approach
@ray.remote
def process_purchase_chunk(chunk_df):
    return chunk_df.groupby(['DATE', 'PURCHASE OFFICER NAME']).agg({...})

# Usage
futures = [process_purchase_chunk.remote(chunk) for chunk in chunks]
results = ray.get(futures)
final_result = pd.concat(results).groupby(['DATE', 'PURCHASE OFFICER NAME']).agg({...})
```

**Verdict**: Ray would help with computation but not with the main bottleneck (I/O operations).

#### AWS Glue ETL Approach
**Better fit for your use case:**

```python
# Glue ETL job structure
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext

def glue_etl_job():
    # 1. Read from S3 (exported Google Sheets data)
    purchase_df = glueContext.create_dynamic_frame.from_catalog(
        database="pullus_db", 
        table_name="purchase_data"
    ).toDF()
    
    # 2. Distributed processing with Spark
    grouped_purchase = purchase_df.groupBy("DATE", "PURCHASE_OFFICER_NAME").agg(
        sum("NUMBER_OF_BIRDS").alias("total_birds"),
        sum("PURCHASED_CHICKEN_WEIGHT").alias("total_chicken")
    )
    
    # 3. Write results back to S3
    grouped_purchase.write.mode("overwrite").parquet("s3://pullus-results/processed/")
```

### Option 2: Batch Processing + Local Optimization (Recommended)

**Immediate scalability without architecture changes:**

```python
class ScalableDiscrepancyAnalyzer(DiscrepancyAnalyzer):
    def read_sheet_data_chunked(self, sheet_id, sheet_name="Sheet1", chunk_size=10000):
        """Process data in chunks to manage memory and API limits"""
        all_data = []
        start_row = 4  # After headers
        
        while True:
            try:
                # Read chunk with range specification
                range_name = f"A{start_row}:Z{start_row + chunk_size - 1}"
                chunk_values = self._api_call_with_retry(
                    lambda: self.gc.open_by_key(sheet_id)
                    .worksheet(sheet_name)
                    .get_values(range_name)
                )
                
                if not chunk_values:
                    break
                    
                # Process chunk immediately to reduce memory usage
                chunk_df = pd.DataFrame(chunk_values, columns=self.headers)
                processed_chunk = self._process_chunk(chunk_df)
                all_data.append(processed_chunk)
                
                start_row += chunk_size
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"Error processing chunk starting at row {start_row}: {e}")
                break
        
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    def process_with_parallel_io(self, purchase_sheet_id, inventory_sheet_id):
        """Use concurrent futures for parallel API calls"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def read_sheet_worker(args):
            sheet_id, sheet_name = args
            return self.read_sheet_data_chunked(sheet_id, sheet_name)
        
        # Parallel reading
        with ThreadPoolExecutor(max_workers=2) as executor:
            tasks = [
                executor.submit(read_sheet_worker, (purchase_sheet_id, "Pullus Purchase Tracker")),
                executor.submit(read_sheet_worker, (inventory_sheet_id, "Pullus Inventory Tracker"))
            ]
            
            purchase_df = tasks[0].result()
            inventory_df = tasks[1].result()
        
        return purchase_df, inventory_df
```

### Option 3: Hybrid Cloud Architecture

**Best of both worlds approach:**

```python
class HybridCloudAnalyzer:
    def __init__(self):
        self.local_analyzer = DiscrepancyAnalyzer()
        self.cloud_processor = CloudProcessor()  # BigQuery, Redshift, etc.
    
    def scale_processing_pipeline(self, purchase_sheet_id, inventory_sheet_id):
        """
        1. Export Google Sheets to cloud warehouse
        2. Process with SQL/Spark in cloud
        3. Import results back to Google Sheets
        """
        
        # Step 1: Export to cloud (one-time bulk operation)
        purchase_data = self._export_to_cloud_warehouse(purchase_sheet_id)
        inventory_data = self._export_to_cloud_warehouse(inventory_sheet_id)
        
        # Step 2: Cloud processing (handles millions of records)
        discrepancy_results = self._process_in_cloud(purchase_data, inventory_data)
        
        # Step 3: Import summary results back to sheets
        self._update_sheets_with_cloud_results(discrepancy_results)
```

## Performance Comparison

| Approach | Dataset Size | Processing Time | Memory Usage | Setup Complexity | Cost |
|----------|-------------|----------------|--------------|------------------|------|
| **Current** | 1K-50K | 5-30 min | Low | Low | Low |
| **Chunked Local** | 50K-500K | 15-90 min | Medium | Low | Low |
| **AWS Glue ETL** | 500K-10M+ | 10-45 min | High (distributed) | High | Medium |
| **Hybrid Cloud** | 100K-10M+ | 20-60 min | Low (local) | Medium | Medium |
| **Full Cloud** | 1M-100M+ | 5-30 min | High (distributed) | High | High |

## Implementation Roadmap

### Phase 1: Immediate Optimizations (1-2 days)
1. **Implement chunked reading** for large datasets
2. **Add parallel I/O** for concurrent sheet reading/writing
3. **Memory optimization** with streaming processing
4. **Enhanced rate limiting** with adaptive backoff

```python
# Quick wins implementation
def optimize_current_architecture(self):
    # 1. Chunked processing
    self.chunk_size = 10000
    
    # 2. Parallel I/O
    self.max_concurrent_requests = 2
    
    # 3. Memory management
    self.enable_streaming_mode = True
    
    # 4. Adaptive rate limiting
    self.adaptive_backoff = True
```

### Phase 2: Hybrid Architecture (1-2 weeks)
1. **Cloud warehouse integration** (BigQuery/Redshift)
2. **Automated export/import pipelines**
3. **SQL-based aggregation queries**
4. **Result caching and incremental updates**

### Phase 3: Full Cloud Migration (2-4 weeks)
1. **AWS Glue ETL jobs** for heavy processing
2. **S3 data lake** for raw and processed data
3. **Lambda functions** for orchestration
4. **CloudWatch** for monitoring and alerting

## Code Examples

### Enhanced Chunked Processing
```python
def process_large_dataset_optimized(self, sheet_id, sheet_name):
    """Optimized processing for large datasets"""
    
    # Configuration
    CHUNK_SIZE = 10000
    MAX_RETRIES = 3
    BATCH_WRITE_SIZE = 1000
    
    # Initialize aggregators
    aggregated_results = defaultdict(lambda: {
        'total_birds': 0,
        'total_chicken_weight': 0,
        'total_gizzard_weight': 0,
        'invoice_numbers': set()
    })
    
    start_row = 4
    processed_rows = 0
    
    while True:
        # Read chunk
        chunk_data = self._read_chunk_with_retry(
            sheet_id, sheet_name, start_row, CHUNK_SIZE, MAX_RETRIES
        )
        
        if not chunk_data:
            break
        
        # Process chunk in streaming fashion
        for row in chunk_data:
            key = (row['DATE'], row['PURCHASE_OFFICER_NAME'])
            aggregated_results[key]['total_birds'] += float(row['NUMBER_OF_BIRDS'])
            aggregated_results[key]['total_chicken_weight'] += float(row['PURCHASED_CHICKEN_WEIGHT'])
            aggregated_results[key]['invoice_numbers'].add(row['INVOICE_NUMBER'])
        
        processed_rows += len(chunk_data)
        start_row += CHUNK_SIZE
        
        # Progress reporting
        if processed_rows % 50000 == 0:
            print(f"Processed {processed_rows:,} rows...")
        
        # Rate limiting
        time.sleep(0.1)
    
    return self._convert_aggregated_to_dataframe(aggregated_results)
```

### Cloud Processing Integration
```python
def setup_cloud_processing(self):
    """Setup cloud processing infrastructure"""
    
    # BigQuery setup
    from google.cloud import bigquery
    
    client = bigquery.Client()
    
    # Create processing SQL
    processing_query = """
    WITH purchase_aggregated AS (
        SELECT 
            DATE,
            PURCHASE_OFFICER_NAME,
            SUM(NUMBER_OF_BIRDS) as total_birds,
            SUM(PURCHASED_CHICKEN_WEIGHT) as total_chicken_weight,
            STRING_AGG(INVOICE_NUMBER, ',') as invoice_numbers
        FROM `pullus.purchase_data`
        GROUP BY DATE, PURCHASE_OFFICER_NAME
    ),
    inventory_aggregated AS (
        SELECT 
            DATE,
            PURCHASE_OFFICER_NAME,
            SUM(NUMBER_OF_BIRDS) as total_birds,
            SUM(INVENTORY_CHICKEN_WEIGHT) as total_chicken_weight
        FROM `pullus.inventory_data`
        GROUP BY DATE, PURCHASE_OFFICER_NAME
    )
    SELECT 
        COALESCE(p.DATE, i.DATE) as date,
        COALESCE(p.PURCHASE_OFFICER_NAME, i.PURCHASE_OFFICER_NAME) as officer,
        p.total_birds as purchase_birds,
        i.total_birds as inventory_birds,
        (p.total_birds - i.total_birds) as bird_difference,
        p.total_chicken_weight as purchase_chicken,
        i.total_chicken_weight as inventory_chicken,
        (p.total_chicken_weight - i.total_chicken_weight) as chicken_difference
    FROM purchase_aggregated p
    FULL OUTER JOIN inventory_aggregated i
        ON p.DATE = i.DATE 
        AND p.PURCHASE_OFFICER_NAME = i.PURCHASE_OFFICER_NAME
    """
    
    return client.query(processing_query)
```

## Monitoring and Alerting

### Performance Metrics to Track
```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'processing_time': [],
            'memory_usage': [],
            'api_calls_count': 0,
            'error_rate': 0,
            'records_processed': 0
        }
    
    def track_processing_performance(self, func):
        """Decorator to track function performance"""
        import psutil
        import time
        
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = func(*args, **kwargs)
                self.metrics['records_processed'] += len(result) if hasattr(result, '__len__') else 0
            except Exception as e:
                self.metrics['error_rate'] += 1
                raise
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                self.metrics['processing_time'].append(end_time - start_time)
                self.metrics['memory_usage'].append(end_memory - start_memory)
            
            return result
        return wrapper
```

## Recommendations

### For Immediate Implementation (Next Sprint)
1. **Start with Option 2 (Batch Processing + Local Optimization)**
   - Lowest risk, immediate benefits
   - Can handle 100K-500K records efficiently
   - Builds on existing architecture

### For Medium-term (Next Quarter)
1. **Implement Hybrid Cloud Architecture**
   - Best balance of performance and complexity
   - Handles 1M+ records efficiently
   - Maintains current workflow familiarity

### For Long-term (Next 6 months)
1. **Consider Full Cloud Migration**
   - Only if processing requirements exceed 10M+ records
   - Requires significant architectural changes
   - Higher operational complexity

## Cost Analysis

| Approach | Development Time | Infrastructure Cost/Month | Maintenance Effort |
|----------|------------------|---------------------------|-------------------|
| **Optimized Local** | 1-2 days | $0 | Low |
| **Hybrid Cloud** | 1-2 weeks | $50-200 | Medium |
| **Full Cloud** | 2-4 weeks | $200-1000 | High |

## Conclusion

The current architecture can be significantly optimized for large datasets without major architectural changes. The chunked processing approach with parallel I/O provides the best immediate return on investment, while the hybrid cloud approach offers the best long-term scalability solution.

Start with local optimizations, then gradually migrate to cloud processing as data volumes and processing requirements grow.

---

*Last updated: September 2024*
*Contact: Technical Architecture Team*