# Agentic Pipeline Report: 20 Newsgroups (4 categories)

The raw newsgroup text still has the email headers in it (`From:`, `Subject:`, `Organization:`, `Lines:`), and these headers affect almost every result below. Term frequencies, class correlations, frequent patterns, PCA and cosine similarity care more about the header and the writing style than the topic.

## 1. Data and basic checks

I loaded four categories with `load_dataset_tool` (`load_dataset_1`). There are 2,257 documents: soc.religion.christian 599, sci.med 594, comp.graphics 584 and alt.atheism 480. `inspect_data_tool` (`inspect_2`) returned a 2,257 x 3 table (`text`, `category`, `category_name`), and every text starts with a raw header. There is no missing text (`missing_3`: 0) and no duplicate (`duplicates_4`: 0 of 2,257). The text length (`describe_5`) is heavily right-skewed: the mean is 1,993 characters, the median is 1,235, and the standard deviation is 3,548, which is larger than the mean. The maximum is 60,713 in comp.graphics. Hence, a few very long posts may dominate the raw counts.

![Text length statistics per category (describe_5)](../plots/凃智中_agentic-pipeline_plot_describe_5.png)

`sample_6` is a random sample of 5 documents with seed 42, so it can be reproduced.

## 2. Document-term matrix, term frequencies and filtering

`build_dtm_tool` (`dtm_7`) built a 2,257 x 35,788 count matrix with 365,886 non-zero entries, so it is 99.547% sparse. I did not remove the stop words. The top 10 terms (`term_freq_8`) are all function words (the 29,825, of 17,660, to 17,222). When I asked for the top 50 (`term_freq_9`), the header words showed up: `from` is 14th (4,877), `edu` is 16th (4,675), `subject` is 30th (2,464), `lines` is 33rd (2,353), `organization` is 38th (2,192) and `com` is 41st (2,172). The first topical word is `god`, at rank 36 (2,216). The agent said the header words only occur about 2,200 to 2,500 times, but this number did not come from any tool, and `from` is about twice of it.

The first 20 columns of the matrix (`dtm_heatmap_10`) are alphabetical numeric tokens like `00` and `000`, so this slice is almost all zeros. However, it still shows how noisy the vocabulary is.

![Heatmap of a 20 x 20 slice of the document-term matrix (dtm_heatmap_10)](../plots/凃智中_agentic-pipeline_plot_dtm_heatmap_10.png)

The variance filter with threshold 0.01 (`variance_filter_11`) keeps 5,354 of the 35,788 terms and removes 30,434. A term which occurs in only one document has variance 0.000443, so this threshold roughly means a term has to appear in about 1% of the documents.

## 3. Class correlation and the feature correlation matrix

For comp.graphics, Pearson (`pearson_filter_12`) and Spearman (`spearman_filter_13`) give very different top terms. The top positive terms of Pearson are `thanks` (0.278), `keywords` (0.233), `looking` (0.213), `vga`, `video`, `card`, `hi`, `host`, `algorithm` and `nntp`. Only four of them are about graphics. `keywords`, `host` and `nntp` come from the header fields, and `thanks`, `hi` and `looking` come from the tone of help requests. Spearman uses the rank of the data, so it returns the topical terms: `graphics` (0.521), `files` (0.317), `image` (0.309), `3d` (0.306), `program`, `file`, `windows`, `ftp` and `format`.

The two top-10 lists only share `thanks` among the positive terms, and share five among the negative terms (`that`, `not`, `people`, `re`, `who`). The agent first counted four, and I pointed out the fifth one. The bar chart of `spearman_filter_13` is sorted by absolute value, so it mixes `graphics` and `files` with negative function words such as `that`, `of` and `not`. The first text description of this chart from the agent listed different terms, so I describe the chart itself here.

![Bar chart of spearman_filter_13 for comp.graphics, sorted by absolute correlation](../plots/凃智中_agentic-pipeline_plot_turn82.png)

The correlation matrix of the 20 highest-variance terms (`featcorr_14`) shows that the function words move together (`the` and `of` 0.944, `the` and `in` 0.896). However, `jpeg` (0.284 with `the`) and `edu` (0.092 with `not`, the lowest pair) behave differently.

![Correlation matrix of the 20 highest-variance terms (featcorr_14)](../plots/凃智中_agentic-pipeline_plot_featcorr_14.png)

## 4. Frequent patterns

I mined comp.graphics (`patterns_15`) and soc.religion.christian (`patterns_16`) with TF-IDF filtering, FP-Growth and a minimum support of 80. comp.graphics has 584 documents, 10,858 surviving terms and 999 patterns. soc.religion.christian has 599 documents, 10,861 terms and 9,699 patterns. In both classes, the strongest patterns are header words: `lines` and `subject` (the support equals the number of documents, 584 and 599), `organization` (560 and 536) and `edu` (320 and 465), and also the combinations of them. Therefore, these patterns describe the message format, not the topic. `edu` also appears in more christian posts (78%) than graphics posts (55%), which is another difference of the headers.

## 5. Dimension reduction, labels and similarity

PCA (`reduce_pca_17`) puts 65.9% of the variance on the first component and 4.3% on the second. Nearly all documents stay in one blob with a few extreme outliers, and the four categories overlap completely. On the raw counts, the first component mostly reflects the document length.

![PCA of the document-term matrix, colored by category (reduce_pca_17)](../plots/凃智中_agentic-pipeline_plot_reduce_pca_17.png)

UMAP (`reduce_umap_18`) only separates the categories partly. comp.graphics stays mostly at the bottom right, sci.med has a small group on the left, and alt.atheism and soc.religion.christian still overlap a lot. The agent predicted much cleaner clusters, but the plot doesn't show that.

![UMAP of the document-term matrix, colored by category (reduce_umap_18)](../plots/凃智中_agentic-pipeline_plot_reduce_umap_18.png)

`binarize_labels_19` gave a 2,257 x 4 one-hot matrix (counts 480, 584, 594, 599). For the cosine similarity, two comp.graphics documents (0 and 1, `cosine_20`) get 0.3532, which is lower than a comp.graphics and a soc.religion.christian document (0 and 4, `cosine_21`) at 0.3738. The raw counts are dominated by headers and function words, so the cosine similarity can't separate the topics here.

## Takeaway

The pipeline can run from start to end, but the data needs to be cleaned first. I would strip the headers and the quoted text, drop the stop words, and use TF-IDF or length normalization before filtering, mining patterns, reducing dimensions or comparing documents. I think the topic structure which Spearman and UMAP hint at would come out more clearly after that, but I did not run a cleaned version.
