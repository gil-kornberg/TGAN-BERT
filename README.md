# TGAN-BERT: Generating Realistic BERT Embeddings with Transformer GAN

GANs have taken the vision community by storm in recent years. 
However, this success has not extended to applications in NLP due to the discreteness of language. 
Meanwhile, the use of pretrained BERT models as a basis for downstream tasks in NLP has obtained state of the art results in many NLP tasks including question answering and language inference, among others. 
There has been at least one attempt by Croce et. al. in GAN-BERT to combine the robust language representation provided by BERT encodings with the propensity for GANs to generate realistic synthetic data.
In GAN-BERT the authors had the intention of improving classification accuracy in low resource settings by leveraging fake BERT embeddings from the generator, with the final produce being the discriminator which doubles as a multi-class classifier. 
The purpose of this paper is to extend the work done in GAN-BERT in two ways: to make the generator and discriminator more robust using transformers, and to evaluate the generator’s ability to generate realistic BERT embeddings similar to real examples drawn from the bAbI dataset. 
Preliminary results indicate that the transformer based GAN architecture is indeed more robust as it exhibits more diverse embeddings than the vanilla MLP-based model, however generated text samples are far from realistic.

I use the bAbI dataset from Facebook Research.
The data is designed as a toy question answering task. 
It contains 20 different story and question types each with 10,000 examples. 
For example: “Mary moved to the bathroom. John went to the hallway. Where is Mary? The bathroom.” 
Preprocessing involved concatenating the stories, which are comprised of approximately 20 short sentences each, into a single line and creating a custom dataset using a Bert tokenizer from HuggingFace and the dataset class provided by PyTorch. 
The final dataset size is approximately 200,000 lines of text.
