function whiten(data,index_vector,R,C,L)
wD = ones(rows(data),columns(data)); %to initialize a matrix equal to the imagedata size with all zeros
for i = 1:rows(index_vector)
wD(index_vector(i,1),:) = data(index_vector(i,1),:); %to store only one segment,rest will remain zeros
endfor
IMG = reshape(wD,R,C,3);
figure;
imshow(IMG);
end
