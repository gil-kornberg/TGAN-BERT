import torch
from torch import nn
from torch.utils.data import Dataset
# from Dataset import load_dataset
from torch.utils.data import DataLoader
import build_dataset
from tqdm.auto import tqdm
from transformers import BertModel, BertConfig
import torch.nn.functional as F
import TransformerGen_withPos as Gen
import TransformerDisc as Disc
import os
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
torch.autograd.set_detect_anomaly(True)

seqLen = 50
embedSize = 768
batch_size = 10
z_dim = (batch_size, seqLen, embedSize)
criterion = nn.BCEWithLogitsLoss()
n_epochs = 12
display_step = 500
inputSize = seqLen * embedSize
outputSize = seqLen * embedSize
device = "cuda"
filepath = '/home/ubuntu/BERT-GAN/BERT_GAN/master/master_train.txt'


configuration = BertConfig(vocab_size=249)
# model = BertModel(config=configuration)
model = BertModel.from_pretrained('bert-base-cased')
model = model.to(device)


dataset = build_dataset.TrainDataset(filepath)
dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


criterion = nn.BCEWithLogitsLoss()


def save_ckpt(epoch, model, model_name, optimizer, lr_scheduler, ckpt_dir, device):
        """
        Save model checkpoint to disk.

        Args:
            epoch (int): Current epoch
            model : Model to save
            lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler for optimizer
            optimizer (torch.optim.Optimizer): Optimizer for model parameters
            device (str): Device where the model/optimizer parameters belong
            model_name (str): Name of model to save
            ckpt_dir (str): Directory to save the checkpoint
        """
        # Unwrap nn.DataParallel module if needed
        try:
            model_class = model.module.__class__.__name__
            model_state = model.to('cpu').module.state_dict()
            print("Unwrapped DataParallel module.")
        except AttributeError:
            model_class = model.__class__.__name__
            model_state = model.to('cpu').state_dict()

        ckpt_dict = {
            'ckpt_info': {'epoch': epoch},
            'model_class': model_class,
            'model_state': model_state,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler,
        }

        ckpt_path = os.path.join(ckpt_dir, f"{model_name}_epoch_{epoch}.pth.tar")
        torch.save(ckpt_dict, ckpt_path)
        model.to(device)
        print(f"Saved {model_name} at epoch {epoch} to {ckpt_path}.")

def get_disc_loss(gen, disc, criterion, realBERTEmbed, z_dim, realNorms, fakeNorms):
    noise = Gen.get_noise(z_dim)
    fakeBERTEmbed = gen(noise).detach()
    fakeBERTEmbed = fakeBERTEmbed.flatten()
    fakePred = disc(fakeBERTEmbed)
    fakeLoss = criterion(fakePred, torch.zeros_like(fakePred))
    realPred = disc(realBERTEmbed.last_hidden_state.flatten(1))
    realLoss = criterion(realPred, torch.ones_like(realPred))
    discLoss = (fakeLoss + realLoss) / 2
    return discLoss


def get_gen_loss(gen, disc, criterion, z_dim):
    noise = Gen.get_noise(z_dim)
    fakeBERTEmbed = gen(noise)
    fakeBERTEmbed = fakeBERTEmbed.flatten()
    fakePred = disc(fakeBERTEmbed)
    genLoss = criterion(fakePred, torch.ones_like(fakePred))
    return genLoss


def lsgan_disc_adversarial_loss(gen, disc, real, z_dim):
    """Least-squares loss, as defined in https://arxiv.org/abs/1611.04076 (Mao et al. 2016)"""
    noise = Gen.get_noise(z_dim)
    fakeBERTEmbed = gen(noise).detach()
    fakeBERTEmbed = fakeBERTEmbed
    # print('fake shape: ', fakeBERTEmbed.shape)
    disc_fake_pred = disc(fakeBERTEmbed)
    disc_fake_adv_loss = torch.mean(disc_fake_pred ** 2)
    # print('real shape: ', real.last_hidden_state.flatten(1).shape)
    disc_real_pred = disc(real.last_hidden_state)
    disc_real_adv_loss = torch.mean((disc_real_pred - 1.) ** 2)
    disc_adv_loss = (disc_fake_adv_loss + disc_real_adv_loss) / 2
    return disc_adv_loss


def lsgan_gen_adversarial_loss(gen, disc, z_dim):
    """Least-squares loss, as defined in https://arxiv.org/abs/1611.04076 (Mao et al. 2016)"""
    noise = Gen.get_noise(z_dim)
    fakeBERTEmbed = gen(noise)
    fakeBERTEmbed = fakeBERTEmbed
    disc_fake_pred = disc(fakeBERTEmbed)
    gen_adv_loss = torch.mean((disc_fake_pred - 1.) ** 2)
    return gen_adv_loss



def hinge_gen_adversarial_loss(gen, disc, z_dim):
    """Hinge loss, as defined in https://arxiv.org/abs/1705.02894v2 (Lim and Ye 2017)"""
    noise = Gen.get_noise(z_dim)
    fakeBERTEmbed = gen(noise)
    fakeBERTEmbed = fakeBERTEmbed.flatten()
    disc_fake_pred = disc(fakeBERTEmbed)
    gen_adv_loss = torch.mean(-disc_fake_pred)
    return gen_adv_loss


def hinge_disc_adversarial_loss(gen, disc, real, z_dim):
    """Hinge loss, as defined in https://arxiv.org/abs/1705.02894v2 (Lim and Ye 2017)"""
    noise = Gen.get_noise(z_dim)
    with torch.no_grad():
    	fakeBERTEmbed = gen(noise).detach()
    fakeBERTEmbed = fakeBERTEmbed.flatten()
    disc_fake_pred = disc(fakeBERTEmbed)
    disc_fake_adv_loss = torch.mean(F.relu(1 + disc_fake_pred))
    disc_real_pred = disc(real.last_hidden_state.flatten(1))
    disc_real_adv_loss = torch.mean(F.relu(1 - disc_fake_pred))
    disc_adv_loss = (disc_fake_adv_loss + disc_real_adv_loss) / 2
    return disc_adv_loss



lr_gen = 0.0001
lr_disc = 0.00001
cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
gen = Gen.Generator().to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr_gen)
disc = Disc.Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr_disc)


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight, gain=1.0)


def gen_weights_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(0.0, 0.01)
        m.bias.data.zero_()
                                    

# gen = gen.apply(weights_init)
# disc = disc.apply(weights_init)

if __name__ == "__main__":

    # torch.multiprocessing.set_start_method('spawn')
    realNorms = []
    fakeNorms = [] 

    for epoch in range(n_epochs):

        pbar = tqdm(dataLoader)
        for batch in pbar:

            if cur_step % 2 == 0:

                with torch.no_grad():

                #print(batch.is_cuda)
                    realEmbedding = model(batch)
            
                # Zero out the gradients before backpropagation
                disc_opt.zero_grad()

                # Calculate discriminator loss
                # disc_loss = get_disc_loss(gen, disc, criterion, realEmbedding, z_dim, realNorms, fakeNorms)

                disc_loss = lsgan_disc_adversarial_loss(gen, disc, realEmbedding, z_dim)
                # disc_loss = hinge_disc_adversarial_loss(gen, disc, realEmbedding, z_dim)
                
                # Calculate LSGAN discriminator loss
                # disc_loss = lsgan_disc_adversarial_loss(gen, disc, realEmbedding, z_dim)

                disc_loss.backward()
                
                # Update optimizer
                disc_opt.step()

                writer.add_scalar("transformer_disc_train_loss", disc_loss.detach().item(), cur_step)

            # Zero out the gradients before backpropagation
            gen_opt.zero_grad()

            # Calculate generator loss
            # gen_loss = get_gen_loss(gen, disc, criterion, z_dim)

            # Calculate LSGAN generator loss
            gen_loss = lsgan_gen_adversarial_loss(gen, disc, z_dim)

            # Calculate Hinge generator loss
            # gen_loss = hinge_gen_adversarial_loss(gen, disc, z_dim)
            
            # Update gradients
            gen_loss.backward()

            # Update optimizer
            gen_opt.step()

            writer.add_scalar("transformer_gen_train_loss", gen_loss.detach().item(), cur_step)
            
            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step

            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step

            cur_step += 1
            pbar.set_description(f"d: {round(disc_loss.item(), 3)}, g: {round(gen_loss.item(), 3)}")
            pbar.update(1)
        
            # Visualization code
            if cur_step % display_step == 0 and cur_step > 0:
                print(f"Epoch {epoch}, step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                mean_discriminator_loss = 0
                mean_generator_loss = 0
        pbar.close()
        # add save
        save_ckpt(epoch, gen, "transformer_pos_lsgan_loss_generator", gen_opt, lr_gen, '/home/ubuntu/BERT-GAN/BERT_GAN/generator_ckpts', device)
    writer.flush()
writer.close()
model.save_pretrained('/home/ubuntu/BERT-GAN/BERT_GAN/bAbI_bert_model')
        
