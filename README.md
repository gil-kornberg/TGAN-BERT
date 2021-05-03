# TGAN-BERT: Toward Generating Realistic BERT Embeddings with Transformer GAN

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

<img width="553" alt="Screen Shot 2021-04-30 at 11 14 48 AM" src="https://user-images.githubusercontent.com/61718766/116667816-4824c780-a9a5-11eb-8c8c-6ebfbceb07df.png">

The training loop involves generating a fake embedding from the generator and a real embedding by passing a data point to the BERT model, and then taking the Least Squares Loss as described in the LSGAN9 paper:

<img width="719" alt="Screen Shot 2021-04-30 at 11 15 59 AM" src="https://user-images.githubusercontent.com/61718766/116667934-72768500-a9a5-11eb-8de7-e0d5102fed49.png">

The augmented approach involved implementing novel generator and discriminator architectures using PyTorch transformers. The generator was replaced by a TransformerEncoder block comprised of eight TransformerEncoderLayers, each with the number of features equal to the embedding size. The discriminator was replaced with four such layers and with an additional final linear layer to output a single value for prediction. In addition, in an effort to further strengthen the generator at the expense of the discriminator, the generator transformer contains an initial positional encoding layer.

Example outputs: "Laurence Laurence Laurel was Laurel was was Laurel was was was seated was seated," "except first in, everything with then and only including," "was everybody with was everybody with everybody with with everybody," "first was most first first was most first first first."

While the model lacks the capacity to generate realistic text, there are several interesting characteristics that merit discussion. For instance, the TGAN generate the example:, “Laurence Laurence Laurel was Laurel was was Laurel was was was seated was seated.” While this is not anything near a sentence, it does have the shades of a sentence that goes something like “Laurence was seated.”

There are several intriguing directions for future investigations. Vincent Liu suggested using a Patch-GAN discriminator and intuitively this seems like a very promising idea. Currently the discriminator is outputting a scalar value indicating how real a 50*768 matrix (50 fake BERT embeddings) is. A Patch-GAN like discriminator seems to suit the task better. There are several ways one could envision strengthening the generator. For instance, I can envision a curriculum training regimen in which the generator could be warm-started by loading pretrained weights from a pretrained BERT model. Another augmentation could involve changing the task from modeling BERT embeddings to modeling static word embeddings like GloVe vectors. Not only are these static, but they have smaller dimensionality. Similarly, I could envision a U-Net like architecture wherein the embeddings are down-sampled so that the generator can be encouraged to learn from more salient information. Finally, it would certainly be preferable to use a BERT model that is pretrained on domain specific data rather than on general text.
