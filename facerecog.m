%% Copyright (c) 2014, Kalyan S Dash
%% I adjust the source code from http://www.mathworks.com/matlabcentral/fileexchange/45750-face-recognition-using-pca/content/facerecog.m

%{
	Using PCA method to classify.
	@param trainingDataPathList : it's array of training data path
	@param testImgPath : it's test data img path
	@return recognized_img : in path format	
	@assumption all the imgs are in the same size
%}
function [ recognized_img ] = facerecog( trainingDataPathList, testImgPath )
% to simplify to calculation : use gray img instead of rgb
imgcount = size(trainingDataPathList , 1);
X = []; %% N feature vector (p dim) in p * N
for i = 1 : imgcount
   X = [X , imreadOneD( trainingDataPathList(i,: ))];
end

m = mean(X,2); % compute the mean of feature vectors
% substract each vector in X with mean in A
A = []; 
for i = 1 : imgcount
    A = [A double(X(:,i))-m ];
end

L = A' * A;
[V,D] = eig(L);

%Select the eigenvector only corresponding eigenvalue > 1
L_eig_vec = [];
for i = 1 : size(V,2);
   if(D(i,i) > 1)
       L_eig_vec = [L_eig_vec V(:,i)];
   end
end

eigenfaces = A * L_eig_vec;

projectimg = [];
for i = 1 : size(eigenfaces,2)
   projectimg = [projectimg eigenfaces' * A(:,i)];
end

testimg = imreadOneD(testImgPath);
testimg = double(testimg) - m ; %substract with mean from the training vectors
projtestimg = eigenfaces' * testimg;

[classto, error] = getNearestIndex(projectimg, projtestimg)
figure;
subplot(1,2,1), subimage(imread(trainingDataPathList(classto,:))), title('classto')
subplot(1,2,2), subimage(imread(testImgPath)),title('test img')
end

%% imreadOneD: function description
function [outputs] = imreadOneD(imgPath)
	img = imreadGray(imgPath);
	[r c] = size(img);
	outputs = reshape(img',r*c,1);
end


%% imreadGray: function description
function [img] = imreadGray(imgPath)
	img = imread(imgPath);
	if (size(img,3) == 3)
		img = rgb2gray(img);
	end

end



%% getNearestIndex: function description
function [index error] = getNearestIndex(projectimg, referece)
	euclide_dist = [];
	for i = 1 : size(projectimg,2)	
		euclide_dist = [euclide_dist (norm(projectimg(:,i)-referece))^2];
	end

	[error index] = min(euclide_dist);
end

