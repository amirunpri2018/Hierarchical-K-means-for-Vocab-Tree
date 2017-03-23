X=[];
Y=[];
count=0;
myDir = '/home/pathak/vision/assign 1/train_set/';
count = count +1  ;      
fnames = dir(myDir) ; 
numfids = length(fnames);     
   for t = 1:numfids
       count
        fullFileName = fullfile(myDir, fnames(t).name);
        if fullFileName(end) == '.'
         continue 
        end
        RGB = imread(fullFileName);
        I = rgb2gray(RGB) ; 
        Points = detectSURFFeatures(I);
        [features,vpts1] = extractFeatures(I,Points);
        csvwrite(strcat('train_points/' , fnames(t).name, '.dat'),features) ;
        close all 
   end
