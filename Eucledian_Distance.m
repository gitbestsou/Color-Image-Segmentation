function D = Eucledian_Distance(trainData,weight,nrows,ncols,number_of_features)
  
  weight = reshape(weight,nrows*ncols,number_of_features);
  %Before this,the weights were in 3D matrix,this converts the 3D matrix into 2D where each column contains the weights of all output neuron for a feature
  D = sum((trainData-weight).^2,2); %column wise sum
  D = sqrt(D); % I guess,this step can be skipped.
  
end