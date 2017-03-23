X=[];
Y=[];
count=0;
myDir = '/home/pathak/vision/assign 1/done/';
folderPattern = fullfile(myDir, '*_*');
folders = dir(folderPattern);
for K = 1:length(folders)
         myFolder = strcat('/home/pathak/vision/assign 1/done/',folders(K).name);
     count = count +1  ;      
     fnames = dir(myFolder) ; 
     numfids = length(fnames);     
   for t = 1:numfids
        fullFileName = fullfile(myFolder, fnames(t).name);
        if fullFileName(end) == '.'
         continue 
        end
        
        RGB = imread(fullFileName);
        I = rgb2gray(RGB) ; 
%         imshow(I);
        hold on ; 
        Points = detectSURFFeatures(I);
%         plot(Points);
        [features,vpts1] = extractFeatures(I,Points);
        X = [X;features];
        close all 
   end
   csvwrite(strcat('done_points/' , folders(K).name , '.dat'),X) ;
   X = [] ; 
   
end
count 