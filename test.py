import random
from data import ImageDetectionsField, TextField, RawField
from data import TaoCaps, DataLoader
import evaluation
from models.transformer import Transformer, Encoder, Decoder, ScaledDotProductAttention, CopyGenerator
import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import jieba

def predict_captions(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
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

    scores, _ = evaluation.compute_scores(gts, gen)
    
    with open('./result/Tao_gts.txt','w') as wf:
        [wf.write('{0},{1}\n'.format(key, value)) for key, value in gts.items()]
   
    with open('./result/Tao_gens.txt','w') as wf:
        [wf.write('{0},{1}\n'.format(key, value)) for key, value in gen.items()]
    return scores


if __name__ == '__main__':
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='TaVD Model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--annotation_folder', type=str)
    args = parser.parse_args()

    print('TaVD Model Evaluation')
    
    # Pipeline for image
    image_field = ImageDetectionsField(detections_path='../TaVD_data/')

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='jieba',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = TaoCaps(image_field, text_field, '../TaVD_data/', 'data/Tao/metadatas')
    train_dataset, val_dataset, test_dataset = dataset.splits
    
    text_field.vocab = pickle.load(open('vocab.pkl', 'rb')) 
    print(len(text_field.vocab))
    
    # Model and dataloaders                             
    encoder = Encoder(3, text_field.vocab.stoi['<pad>'], attention_module=ScaledDotProductAttention)
    decoder = Decoder(len(text_field.vocab), 30, 3, text_field.vocab.stoi['<pad>'])
    copy_generator = CopyGenerator(len(text_field.vocab), 1024)
    
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder, copy_generator).to(device)
    
    data = torch.load('../TaVD_data/TAVD.pth')
    model.load_state_dict(data['state_dict'])

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    scores = predict_captions(model, dict_dataloader_test, text_field)
    print(scores)

