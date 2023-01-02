import timm

f = open("/home/image-retrieval/ndir_simulated/model_list.txt", 'w', encoding='utf-8')
avail_pretrained_models = timm.list_models(pretrained=True)
for data in avail_pretrained_models:
    f.write(data+"\n")     
f.close
