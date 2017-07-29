clear;clc;close all;
string = input('Input the image file name --->> ','s');
acIMG = imread(string); %to take the image in a matrix
A = reshape(acIMG,size(acIMG,1)*size(acIMG,2),size(acIMG,3)); %reshaping the matrix in a 2D matrix where each row contains the R,G,B values for one pixel
csvwrite('imagedata.txt',A); %the matrix will be stored in a file named imagedata.txt
data = load('imagedata.txt'); 
data = normalize(data); %normalizing the matrix,for each feature
number_of_segments = input('How many color segments? -> '); %Taking the number of color segments of the image you want 
IMG = reshape(data,size(acIMG,1),size(acIMG,2),size(acIMG,3)); %Just to show the image of which the filename you have entered,to ensure everything is right upto now
imshow(IMG); %showing the image
title('Actual Image');
tic;
number_of_features  = columns(data); 
size_of_lattice = 10;  % size_of_lattice is the size of the output node.size_of_lattice=n means that n*n number of output neurons are in the lattice
number_of_epochs = 25; % the number of iterations you want to run the algorithm
n = rows(data);        % number of training examples
eta0 = 0.1;            % initial learning rate 
etadecay = number_of_epochs; % exponential decay rate of learning rate
sigma0 = size_of_lattice;    % initial size of the topological neighbourhood equal to the radius of the lattice
sigma_decay = number_of_epochs/log(sigma0);
nrows = size_of_lattice;ncols = size_of_lattice;
weight = rand(nrows,ncols,number_of_features); %weights are being initialized with random values.
W = weight;                                                %weight(:,:,i)gives the weight of all the output neurons attched to the input neuron corresponding to the input feature number i
[x y] = meshgrid(1:nrows,1:ncols);

for time = 1:number_of_epochs

  eta = eta0 * exp(-time/etadecay); % The learning rate for the current epoch
  sigma = sigma0 * exp(-time/sigma_decay); % The topological neighbourhood shrinks with each epoch
  width = ceil(sigma); % width is basically the measure of topological neighbourhood which is equal to sigma
  
  for i_train = 1:n % now to update the weights for each input training example,n is the number of training examples
  
    trainData = data(i_train,:); % trainData is basically one training data example,one input vector
    
    dist = Eucledian_Distance(trainData,weight,nrows,ncols,number_of_features); % Calculate the Eucledian Distance between training vector and each output neuron
    
    [~,BMU_index] = min(dist);
    BeMU = BMU_index;
    [BMU_row BMU_col] = ind2sub([nrows ncols],BMU_index);
    h = exp(-(((x-BMU_col).^2)+((y-BMU_row).^2))/(2*sigma*sigma)); % Generate a Gaussian function centered on the location of the best matching unit,based on this,weights will be updated
    
    % Determine the boundary of local neighbourhood
    fromrow = max(1,BMU_row - width);
    torow   = min(BMU_row + width,nrows);
    fromcol = max(1,BMU_col - width);
    tocol   = min(BMU_col + width,ncols);
    
    % Get the neighbouring neurons and determine the size of the neighbourhood
    weight_of_neighbourNeurons = weight(fromrow:torow,fromcol:tocol,:);
    area = size(weight_of_neighbourNeurons);
    
    % Transform the training vector and the Gaussian function into multi-dimensional to facilitate the computation of the neuron weights update
    TRAINDATA = reshape(repmat(trainData,area(1)*area(2),1),area(1),area(2),number_of_features);                   
    H = repmat(h(fromrow:torow,fromcol:tocol),1,1,number_of_features);
    
    % Update the weights of the neurons that are in the neighbourhood of the bmu
    weight_of_neighbourNeurons = weight_of_neighbourNeurons + eta .* H .* (TRAINDATA - weight_of_neighbourNeurons);
    % Put the new weights of the BMU neighbouring neurons back to the entire SOM weights
    weight(fromrow:torow,fromcol:tocol,:) = weight_of_neighbourNeurons;
  endfor
endfor

w = reshape(weight,size_of_lattice^2,number_of_features);
for i = 1:rows(data)
A = sum((w-data(i,:,:)).^2,2);
[m(i) BMU(i)] = min(A);
endfor


for s = 1:size_of_lattice^2
node_mass(s,1) = sum(BMU==s); %node_mass indicates that how many data points are there in each node
%printf("Number of elements in node %d = %d\n",s,val);
endfor

BMU = BMU';

for i = 1:size_of_lattice^2
p = 1;
  for j = 1:rows(BMU)
    if(BMU(j,1)==i)
    index(i,p) = j;
    p++;
    endif
  endfor
endfor

%-------The following part is the clustering part-------------------------------
K = number_of_segments; % Specifying the k in k means
clustering_maxIter = 100; %number of iterations for kmeans clustering
clusterData = w; % The weights after applying SOM is the data to cluster
%initialClusterIndex = ceil(size(clusterData,1)*rand(1,K));
initialClusterIndex = randperm(rows(clusterData))(1:K);

for i = 1:K
  cluster(i,:) = clusterData(initialClusterIndex(i),:);  %k number of cluster center is taken,which are actually any random data points
endfor

for j = 1:clustering_maxIter
  count = zeros(K,1); %to count the number of clusters in every iterations
  clusterSum = zeros(K,size(clusterData,2));
  for i = 1:size(clusterData,1)
      for k = 1:K
          d(k,:) = distance(clusterData(i,:),cluster(k,:)); %to calculate the distance from the data points to present cluster centers 
      endfor 
      for k = 1:K
         if(min(d) == d(k,:)) 
           clusterIndex(i,:) = k; %assigning the data example to a particular cluster center
           clusterSum(k,:) = clusterSum(k,:) + clusterData(i,:);
           count(k,:)++;
         endif  
      endfor    
  endfor
  for k = 1:K
    cluster(k,:) = clusterSum(k,:)/count(k,:); %For next iteration,cluster center is changed to the mean of the data points for individual clusters
  endfor
  
endfor

for i = 1:K
p = 1;
  for j = 1:rows(clusterIndex)
    if(clusterIndex(j,1) == i)
      cIndex(i,p) = j;
      p++;
    endif
  endfor
endfor  

for i = 1:K
  C = cIndex(i,:)';
  C(~any(C,2),:) = [];
  p = 1;
  for j = 1:rows(C)
    I = index(C(j,1),:)';
    I(~any(I,2),:) = [];
    for k = 1:rows(I)
      D(i,p) = I(k,1);
      p++;
    endfor
  endfor
endfor

for region = 1:K
index_vector = D(region,:)';
index_vector(~any(index_vector,2),:) = [];
whiten(data,index_vector,size(acIMG,1),size(acIMG,2),size(acIMG,3));
endfor
toc;
