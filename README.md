# THESIS-QUADB
Sentiment-Based Detection of Customer Dissatisfaction in Code-Switched Taglish E-Commerce Reviews Using XLM-RoBERTa

1. It reads a review and decides if the customer is negative, neutral,
   positive, or mixed about the product.
2. For every review it marks as negative, it explains why the customer is
   unhappy, sorted into a fixed set of dissatisfaction categories.

Stage 1 is sentiment classification and stage 2 is dissatisfaction classification ata

We'll be using kaggle for training cause Im not sacrificing my pc to thesis

so far sa src, our preprocess script cleans the text by lowercasing everything, reduce or remove repeated characters kasi bobo mga pilipino, may list of formal tagalog words na chinecheck, bale 'sya' to 'siya' and more, and drop pag blanko ung textbox ng review'

iniisip pa kung iimplement ung spam or irrelevant reviews or yung mga review na may number or emojis pagiisipan pa ig.

