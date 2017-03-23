# Hierarchical-K-means-for-Vocab-Tree

Paper closey followed https://arxiv.org/pdf/1608.01807v1.pdf

first assignment Vision Cs6980

works well for the images, similar to that of training, 
   
   simple testing data : ~5 for top 10 results  
   
   Hard Testing Data having cropped image : poor result 1/10 
   
   Very Hard Testing data : cropped image placed against lot of noise : 0 / 10 


Lp norm distance used for calculation : best results for p = 0.5, then  p= 1 and poor for p = 2

manual tf-idf performed better for some cases. 

