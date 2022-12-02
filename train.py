import random
import jieba
from data import ImageDetectionsField, TextField, RawField
from data import TaoCaps, DataLoader
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer import Transformer, Encoder, Decoder, ScaledDotProductAttention, CopyGenerator
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 3333
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

def evaluate_loss(model, dataloader, loss_fn, text_field, dual):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (detections, object_feats, geo_weights, sen_weights, ocr_list, ocr_vec, shot_mask, captions) in enumerate(dataloader):
                detections, captions = detections.to(device), captions.to(device)
                object_feats, geo_weights, sen_weights = object_feats.to(device), geo_weights.to(device), sen_weights.to(device)
                ocr_list, ocr_vec = ocr_list.to(device), ocr_vec.to(device)
                shot_mask = shot_mask.to(device)
                out, constr_out, context_dec = model(detections, object_feats, geo_weights, sen_weights, captions, ocr_list, ocr_vec, shot_mask)
                
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous() 
                constr_out = constr_out[:, :-1].contiguous()
                context_dec = context_dec[:, :-1].contiguous()
            
                loss_con, pad_index = loss_Contrastive(constr_out, captions, ocr_list)

                context_dec = context_dec.type(torch.float32)
                context_dec = (torch.diagonal(context_dec, dim1=-2, dim2=-1)*pad_index).reshape(-1,1)
                tmie_gth_index = torch.zeros_like(context_dec).type(torch.long).squeeze(dim=1)
                loss_time = loss_fn( context_dec, tmie_gth_index)
                
                loss_cap = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))

                loss = loss_cap + dual*loss_time + dual*loss_con
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss_cap=loss_cap.item() / (it + 1), loss_con=loss_con.item() / (it + 1), loss_time=loss_time.item() / (it + 1), loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss

def evaluate_metrics(model, dataloader, text_field, e):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            detections, object_feats, geo_weights, sen_weights = images[0].to(device), images[1].to(device), images[2].to(device), images[3].to(device)
            ocr_list, ocr_vec = images[4].to(device), images[5].to(device)
            shot_mask = images[6].to(device)
            with torch.no_grad():
                out, _ = model.beam_search(detections, object_feats, geo_weights, sen_weights, ocr_list, ocr_vec, shot_mask,
                                           30, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen_i = ' '.join([k for k in jieba.cut(gen_i.replace(' ',''))])
                gen['%d_%d' % (it, i)] = [gen_i, ]

                gts['%d_%d' % (it, i)] = []
                for t in gts_i:
                    t = ' '.join([k for k in jieba.cut(t.replace(' ',''))])
                    gts['%d_%d' % (it, i)].append(t)
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)

    scores, _ = evaluation.compute_scores(gts, gen)

    return scores

def loss_Contrastive(constr_out, captions_gt, ocr_list):
    constr_out = constr_out[:,:,3].type(torch.float32)
    ocr_index = torch.ones_like(captions_gt)
    pad_index = torch.zeros_like(captions_gt)
    for sen in range(0,captions_gt.size(0) ):
        for tok in range(0,len(captions_gt[sen])):
            # <eos> is 3
            if captions_gt[sen][tok] == 3:
                ocr_index[sen][tok+1:] = 0
                break
            if captions_gt[sen][tok] in ocr_list[sen].type(torch.long):
                ocr_index[sen][tok] = -1
                pad_index[sen][tok] = 1
    gth_index = torch.zeros_like(ocr_index.view(-1))
    
    loss_con = loss_fn( (constr_out*ocr_index).view(-1,1),gth_index)
    return loss_con, pad_index

def train_xe(model, dataloader, optim, text_field, dual):
    # Training with cross-entropy
    model.train()
    scheduler.step()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, object_feats, geo_weights, sen_weights, ocr_list, ocr_vec, shot_mask, captions) in enumerate(dataloader):
            detections, captions = detections.to(device), captions.to(device)
            object_feats, geo_weights, sen_weights = object_feats.to(device), geo_weights.to(device), sen_weights.to(device)
            ocr_list, ocr_vec = ocr_list.to(device), ocr_vec.to(device)
            shot_mask = shot_mask.to(device)
            out, constr_out, context_dec = model(detections, object_feats, geo_weights, sen_weights, captions, ocr_list, ocr_vec, shot_mask)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            constr_out = constr_out[:, :-1].contiguous()
            context_dec = context_dec[:, :-1].contiguous()
            
            loss_con, pad_index = loss_Contrastive(constr_out, captions_gt, ocr_list)
            
            context_dec = context_dec.type(torch.float32)
            context_dec = (torch.diagonal(context_dec, dim1=-2, dim2=-1)*pad_index).reshape(-1,1)
            tmie_gth_index = torch.zeros_like(context_dec).type(torch.long).squeeze(dim=1)
            loss_time = loss_fn( context_dec, tmie_gth_index)
            
            loss_cap = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
            loss = loss_cap + dual*loss_time + dual*loss_con
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss
            
            pbar.set_postfix(loss_cap=loss_cap.item() / (it + 1), loss_con=loss_con.item() / (it + 1), loss_time=loss_time.item() / (it + 1), loss=running_loss / (it + 1))
            pbar.update()
            scheduler.step()

    loss = running_loss / len(dataloader)
    return loss

if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='TaVD')
    parser.add_argument('--exp_name', type=str, default='TaVD')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    args = parser.parse_args()
    print(args)

    print('TaVD Training')

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path='../TaVD_data/')

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='jieba',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = TaoCaps(image_field, text_field, '../TaVD_data/', 'data/Tao/metadatas')
    train_dataset, val_dataset, test_dataset = dataset.splits
    # if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
    #     print("Building vocabulary")
    #     text_field.build_vocab(train_dataset, val_dataset, test_dataset, min_freq=1)
    #     pickle.dump(text_field.vocab, open('vocab_%s.pkl' % args.exp_name, 'wb'))
    # else:
    #     text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))
    text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))
    print(len(text_field.vocab))
    
    encoder = Encoder(3, text_field.vocab.stoi['<pad>'], attention_module=ScaledDotProductAttention)
    decoder = Decoder(len(text_field.vocab), 30, 3, text_field.vocab.stoi['<pad>'])
    copy_generator = CopyGenerator(len(text_field.vocab), 1024)
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder, copy_generator).to(device)

    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})


    def lambda_lr(s):
        lr = 5e-5
        for i in range(s // (4659)):
            lr = lr * 0.8
        s += 1
        return lr
    # Initial conditions
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    use_rl = False
    best_cider = .0
    patience = 0
    start_epoch = 0
    dual = 0.001

    print("Training starts")
    for e in range(start_epoch, start_epoch + 20):
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                      drop_last=True)
        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5, shuffle=True,
                                           num_workers=args.workers)
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5)
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5)
        
        
        train_loss = train_xe(model, dataloader_train, optim, text_field, dual)
        writer.add_scalar('data/train_loss', train_loss, e)
        
        # Validation loss
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field, dual)
        writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, text_field, e)
        print("Validation scores", scores)
        val_cider = scores['CIDEr']
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)

        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            best = True

        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'use_rl': use_rl,
        }, 'saved_models/TaVD/%s.pth' % e)
        
        if best:
            copyfile('saved_models/TaVD_0330/%s.pth' % e, 'saved_models/TaVD/%s_best.pth' % args.exp_name)
