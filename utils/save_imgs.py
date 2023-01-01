    ############## quality analysis #####################
def save_
    #root_path = '/home/image-retrieval/ndir_simulated/'
    root_path = os.getcwd()
    q_img_path_list = []
    db_img_path_list = []
    res_save_dir =  '/home/image-retrieval/ndir_simulated/res_save_dir'
    # #q_img_path = '/home/image-retrieval/ndir_simulated/01.q_origin/'
    # #db_img_path = '/home/image-retrieval/ndir_simulated/01.db_new/'
    data_types = ['01.q_origin', '01.db_new']
    for data_type in data_types:
        if data_type == '01.q_origin':
            q_img_paths = os.listdir(os.path.join(root_path, data_type))
            q_img_paths = [os.path.join(root_path, data_type, img) for img in q_img_paths]
            q_img_path_list = natsort.natsorted(q_img_paths)

        else:     
            db_img_paths = os.listdir(os.path.join(root_path, data_type))
            db_img_paths = [os.path.join(root_path, data_type, img) for img in db_img_paths]
            db_img_path_list = natsort.natsorted(db_img_paths)

    print(f'q_img_path_list is {q_img_path_list}')
    #print(f'db is {db_img_path_list}')
    
    #query_idx = 0 ##rank1 #677 #42 #41 #221(-) #1205 #222 #83 #19 #7 #1224 ##rank3 #2(-) #11(-) ##etc #341(같은 폰트)
    topk = 5

    fig = plt.figure(figsize=(10,10)) # rows*cols 행렬의 i번째 subplot 생성
    rows = topk
    cols = 2

    # 하얀 바탕으로
    fig.patch.set_facecolor('xkcd:white')
    # 지정한 query 1개의 topk개의 db images
    for q_idx, each_q_path in enumerate(q_img_path_list):
        query_img = cv2.imread(each_q_path)
        ax = fig.add_subplot(rows, cols, 1)
        ax.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"Query:{os.path.basename(each_q_path[:-4])}", fontdict={'fontsize': 10})
        for tmp_idx, db_idx in enumerate(I[q_idx][:topk]): # [  99   66 3290]
            print(tmp_idx, db_idx)
            print(f"matched db_img_path is {db_img_path_list[db_idx]}")
            matched_db_img = cv2.imread(db_img_path_list[db_idx]) #쿼리 이미지 한번 돌때 각각의 top 3 이미지 가가각 
            ax = fig.add_subplot(rows, cols, 2*(tmp_idx+1))
            ax.imshow(cv2.cvtColor(matched_db_img, cv2.COLOR_BGR2RGB))
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"DB_top{tmp_idx+1}", fontdict={'fontsize': 10})
            fig.savefig(res_save_dir + '/' +str(os.path.basename(each_q_path[:-4])) +'.jpg')
        fig.clf()
        print("==="*100)
