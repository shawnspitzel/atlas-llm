# 2.1  - Encoding
### a) Nothing
### b) The __repr__ representation outputs it as a string \x00, where as when printed normally, we just get nothing.
### c) When put in text, chr(0) doesn't show up at all and makes no impact on whatever string it's concatenated with.
# 2.2 - UTF-8
### a) UTF-8 is much more compact, using only 1-4 bytes per character, whereas UTF-16 and UTF-32 use 2 and 4 at their lowest respectively.
### b) The reason why it's wrong is because when given multi-byte characters, the iteration logic breaks since it expects only one-byte decoding at a time. Given any multi-byte character like ん will result in it raising an error.
### c) んん will not decode in this implementation.

# 2.5 - Tokenizer Training
### a) It took ~8 hours to train the tokenizer on the tinystories dataset. This can most likely be improved through algorithmic choices and multiprocessing. The longest token in the vocabulary is " enthusiastically".
### b) By far, rebuilding the corpus took the most time out of any other part of the training process, taking up 64% of total compute time. This can most likely be optimized by only rewriting affected pairs rather than rewriting the entire corpus on every merge.
