# LLM Training Data Generation - Gap Analysis

**⚠️ NOTE:** This analysis has been consolidated into `tiger_semantic_id/AGENTS.md` → "LLM Training Data Gap Analysis" section.

**Goal:** Match reference implementation ([semantic-ids-llm](https://github.com/eugeneyan/semantic-ids-llm)) data generation exactly (~4.2M examples from Amazon 2023 Video Games dataset).

**Date:** 2025-11-20
**Status:** ✅ ROOT CAUSE IDENTIFIED - See AGENTS.md for details

**Quick Summary:**
- **Issue:** Generating only 2.66M examples (37% below target)
- **Root Cause:** User interaction filter (min_user_interactions ≥ 5) removes 91% of items
- **Solution:** Reduce threshold to 3 or 2 in data_preparation.ipynb
- **Full Analysis:** See `AGENTS.md` → "LLM Training Data Gap Analysis"

---

## Current Results vs Reference

### Dataset Statistics

| Metric | Current | Reference | Gap | Status |
|--------|---------|-----------|-----|--------|
| **Items** | 42,100 | 66,133 | -36% | ❌ Too low |
| **Users** | 101,409 | 78,643 | +29% | ⚠️ Too high |
| **Avg Sequence Length** | 6.18 | 6.5 | -5% | ✅ Close |
| **Total Examples** | 2,662,865 | 4,200,000 | -37% | ❌ Too low |

### Training Examples Breakdown

| Type | Description | Current | Reference | Gap | Status |
|------|-------------|---------|-----------|-----|--------|
| **A** | SID → Text (title, desc, features, category) | 283,601 | ~318,000 | -11% | ⚠️ Low |
| **B** | Text → SID (reverse mapping) | 364,476 | ~478,000 | -24% | ❌ Low |
| **C** | Next-item prediction (sequential) | 1,271,487 | ~1,400,000 | -9% | ⚠️ Low |
| **D** | Semantic understanding (hierarchy) | 60,000 | ~63,000 | -5% | ✅ Close |
| **E** | Co-purchase patterns (collaborative) | 683,301 | 2,569,685 | -73% | ❌ Very low |
| **Total** | | 2,662,865 | 4,200,000 | -37% | ❌ Too low |

---

## Root Cause Analysis

### Primary Issue: Insufficient Item Count

**Problem:** We start with only 80,840 items in `item_metadata.json`, but reference starts with 137,269 items from the same dataset.

**Filtering Pass Rates:**
- **Our dataset:** 42,100 / 80,840 = **52.1% pass rate**
- **Reference:** 66,133 / 137,269 = **48.2% pass rate**

**Conclusion:** Our pass rate is actually HIGHER than reference, but we're missing ~56K items (41%) from the raw dataset before filtering even begins.

### Secondary Issue: Type E Co-purchase Severely Undersized

**Problem:** Type E generates only 683,301 examples (26% of expected 2.57M).

**Analysis:**
- Current: 683,301 / 42,100 items = **16.2 examples per item**
- Reference: 2,569,685 / 66,133 items = **38.9 examples per item**

Even accounting for fewer items, we should generate ~1.6M examples (42,100 × 38.9), not 683K.

**Potential causes:**
1. Filtered dataset has sparser co-occurrence patterns
2. Sequences are shorter or less dense after filtering
3. Co-occurrence window or exhaustive mode not working as expected

---

## Hypotheses

### Hypothesis 1: Incomplete Raw Dataset Loading ⭐ MOST LIKELY

**Theory:** `TIGER_SemanticID.ipynb` is not extracting metadata for all items in the raw Amazon 2023 Video Games dataset.

**Evidence:**
- We have 80,840 items, reference has 137,269 items (41% missing)
- Both use same dataset (Amazon Reviews 2023 - Video Games)
- Missing items likely explains missing examples across all types

**Investigation needed:**
1. Check raw dataset file size and row count
2. Verify `load_meta_df()` is processing all rows
3. Check if there's unintentional filtering in metadata extraction
4. Confirm we downloaded the correct/complete dataset file

**Files to check:**
- `notebooks/tiger_semantic_id/TIGER_SemanticID.ipynb` - metadata extraction cell
- `tiger_semantic_id/src/data.py` - `load_meta_df()` function

### Hypothesis 2: Description Extraction Failure

**Theory:** Description field extraction from Amazon 2023 format is incomplete or broken.

**Evidence:**
- Type A/B examples are lower than expected
- Filtering requires description ≥100 chars, but extraction might be faulty
- Amazon 2023 format stores descriptions as lists that need joining

**Investigation needed:**
1. In `TIGER_SemanticID.ipynb`, check description extraction:
   ```python
   items_with_desc = items['description'].notna().sum()
   desc_lengths = items['description'].str.len().describe()
   ```
2. Verify `load_meta_df()` properly joins description lists into strings
3. Check for empty/null descriptions that should have content

**Code location:**
- `tiger_semantic_id/src/data.py:199-205` - Description joining logic

### Hypothesis 3: Wrong Dataset Variant

**Theory:** We downloaded a filtered/sampled version of Video Games dataset, not the full version.

**Evidence:**
- Significant item count discrepancy (80K vs 137K)
- URL used: `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Video_Games.jsonl.gz`

**Investigation needed:**
1. Verify dataset URL is correct
2. Check downloaded file size matches expected
3. Check if there are multiple Video Games dataset versions (5-core, full, etc.)

### Hypothesis 4: Sequence Filtering Too Aggressive

**Theory:** Our sequence filtering removes too many user sequences, reducing co-occurrence density.

**Evidence:**
- We have 101K users after filtering (29% more than reference's 79K)
- But fewer co-purchase examples (27% of expected)
- Suggests sequences are shorter/sparser

**Investigation needed:**
1. Check sequence length distribution before/after filtering
2. Compare user retention rate with reference
3. Analyze co-occurrence matrix sparsity

---

## Metadata Filtering Applied

**Reference implementation requires:**
- Title length ≥ 20 characters
- Description length ≥ 100 characters
- Items without BOTH are excluded
- User sequences must have ≥ 3 items after item filtering

**Implementation location:**
- `notebooks/tiger_semantic_id/TIGER_SemanticID_LLM_finetune.ipynb` - Cell 9

**Results:**
- ✅ Filtering logic matches reference exactly
- ✅ Pass rate (52%) similar to reference (48%)
- ❌ Starting item count too low (80K vs 137K)

---

## Code Changes Implemented

### 1. Exhaustive Co-purchase Mode
**File:** `tiger_semantic_id/src/llm/build_sid_dialogs.py`

**Change:** Modified `create_copurchase_dialogs()` to generate examples for ALL co-occurring items when `examples_per_item > top_k`:

```python
if examples_per_item > top_k or top_k == -1:
    # Exhaustive mode: use ALL co-occurring items
    co_items = cooccurrence[item_id]
else:
    # Limited mode: use only top K
    co_items = cooccurrence[item_id][:top_k]
```

**Status:** ✅ Working (generating >10 examples per item)

### 2. Metadata Filtering
**File:** `notebooks/tiger_semantic_id/TIGER_SemanticID_LLM_finetune.ipynb` - Cell 9

**Change:** Added filtering step before dialog generation:
- Filters items by title ≥20, description ≥100
- Filters user sequences to only include valid items
- Removes sequences with <3 items
- Saves filtered artifacts to LLM_DIR

**Status:** ✅ Working, but upstream data insufficient

### 3. Utility Functions
**File:** `tiger_semantic_id/src/data.py`

**Added:**
- `filter_items_by_metadata()` - Filters items by metadata requirements
- `filter_sequences_by_valid_items()` - Filters user sequences

**Status:** ✅ Working correctly

---

## Next Steps (Priority Order)

### 1. ⭐ URGENT: Investigate Raw Dataset Size
**Action:** Check if we're loading the complete Amazon 2023 Video Games dataset.

**Steps:**
1. In `TIGER_SemanticID.ipynb`, add diagnostics:
   ```python
   # After load_meta_df
   print(f"Raw metadata rows: {len(meta)}")
   print(f"Expected: ~137,269 for Video Games full dataset")

   # Check file size
   import os
   file_size = os.path.getsize(meta_path) / (1024**2)  # MB
   print(f"Metadata file size: {file_size:.1f} MB")
   ```

2. Verify dataset URL is correct
3. Check for parsing errors or row skipping in `load_meta_df()`

**Expected outcome:** Find why we have only 80K items instead of 137K

### 2. Verify Description Extraction
**Action:** Check if descriptions are being extracted properly from Amazon 2023 format.

**Steps:**
1. In `TIGER_SemanticID.ipynb`, after metadata extraction:
   ```python
   print("\nDescription statistics:")
   has_desc = items['description'].notna()
   print(f"Items with description: {has_desc.sum()} / {len(items)}")

   desc_lens = items[has_desc]['description'].str.len()
   print(f"Avg description length: {desc_lens.mean():.1f} chars")
   print(f"Description length quantiles:")
   print(desc_lens.quantile([0.25, 0.5, 0.75, 0.9]))
   ```

2. Check `load_meta_df()` description joining logic
3. Sample a few items to verify descriptions look correct

**Expected outcome:** Confirm descriptions are extracted and formatted properly

### 3. Analyze Co-occurrence Density
**Action:** Understand why Type E is so low even with exhaustive mode.

**Steps:**
1. Add diagnostics to co-occurrence matrix building:
   ```python
   # In build_sid_dialogs.py, after building co-occurrence matrix
   avg_cooccurrences = np.mean([len(items) for items in cooccurrence.values()])
   print(f"Average co-occurrences per item: {avg_cooccurrences:.1f}")
   ```

2. Compare with reference's co-occurrence statistics
3. Check if filtered sequences are too sparse

**Expected outcome:** Understand co-occurrence sparsity and whether filtering is too aggressive

### 4. Consider Temporary Workaround
**Action:** Relax filtering thresholds to increase item count.

**Change in Cell 9:**
```python
MIN_TITLE_LENGTH = 10      # Relaxed from 20
MIN_DESCRIPTION_LENGTH = 50  # Relaxed from 100
```

**Trade-off:** Get more data but lower quality metadata

**Status:** Last resort if root cause can't be fixed

---

## Success Criteria

To match reference implementation, we need:

✅ **Dataset:**
- ~66K items (after filtering)
- ~79K users (after filtering)
- Avg 6.5 items per sequence

✅ **Training Examples:**
- Type A: ~320K examples
- Type B: ~480K examples
- Type C: ~1.4M examples
- Type D: ~60K examples
- Type E: ~2.6M examples
- **Total: ~4.2M examples**

---

## References

- **Reference Implementation:** https://github.com/eugeneyan/semantic-ids-llm
- **Reference Data Prep Notebook:** `notebooks/01-prep-items-and-sequences.ipynb`
- **Amazon 2023 Dataset:** https://amazon-reviews-2023.github.io/
- **Our Implementation:** `notebooks/tiger_semantic_id/TIGER_SemanticID_LLM_finetune.ipynb`

---

## Change Log

| Date | Change | Impact |
|------|--------|--------|
| 2025-11-20 | Added exhaustive co-purchase mode | Type E: 688K → 1.38M (unfiltered) |
| 2025-11-20 | Added metadata filtering (title≥20, desc≥100) | Items: 80K → 42K, Total: 4.66M → 2.66M |
| 2025-11-20 | Fixed artifact copying in filtering cell | Resolved missing file errors |
| 2025-11-20 | Created LLM_DATA.md | - |
