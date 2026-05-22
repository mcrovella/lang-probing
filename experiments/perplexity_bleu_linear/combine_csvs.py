import pandas as pd

2# Load the new data files
perplexity_df = pd.read_csv('/projectnb/mcnet/jbrin/lang-probing/outputs/perplexity_bleu_linear/bleu_and_ppl/perplexity_results_llama.csv')
bleu_df = pd.read_csv('/projectnb/mcnet/jbrin/lang-probing/outputs/perplexity_bleu_linear/bleu_and_ppl/bleu_results_llama.csv')

# Prepare the perplexity dataframe for merging on the 'src' column
src_perp = perplexity_df.rename(columns={
    'Language': 'src', 
    'Perplexity': 'src_perplexity'
})

# Merge to add src_perplexity_error
combined_df = bleu_df.merge(src_perp, on='src', how='left')

# Prepare the perplexity dataframe for merging on the 'tgt' column
tgt_perp = perplexity_df.rename(columns={
    'Language': 'tgt', 
    'Perplexity': 'tgt_perplexity'
})

# Merge to add tgt_perplexity_error
combined_df = combined_df.merge(tgt_perp, on='tgt', how='left')

# Save the combined dataframe to the requested CSV filename
combined_df.to_csv('/projectnb/mcnet/jbrin/lang-probing/outputs/perplexity_bleu_linear/bleu_and_ppl/combined_results_llama.csv', index=False)

print(combined_df.head())