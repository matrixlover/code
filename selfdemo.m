function selfdemo
% ====== Self demo using IRIS dataset
% ====== 1. Plot IRIS data after LDA for dimension reduction to 2D
load iris.dat

feaNorm = NormalizeFea(iris(:,1:4),0);
fea = feaNorm;
% iris(:,1:4) ./ repmat(max(1e-10,feaNorm),1,4);

options = [];
options.NeighborMode = 'KNN';
options.WeightMode = 'Cosine';
options.k = 3;

W = constructW(fea,options);

% [LaplacianScore] = feval(mfilename,iris(:,1:4),W);
Score = LaplacianScore(fea, W);
[junk, index] = sort(Score, 'descend');

index1 = find(iris(:,5)==1);
index2 = find(iris(:,5)==2);
index3 = find(iris(:,5)==3);
figure;
plot(iris(index1, index(1)), iris(index1, index(2)), '*', ...
     iris(index2, index(1)), iris(index2, index(2)), 'o', ...
     iris(index3, index(1)), iris(index3, index(2)), 'x');
legend('Class 1', 'Class 2', 'Class 3');
title(sprintf('IRIS data onto the %s and %s feature (Laplacian Score)', num2str(index(1)), num2str(index(2))));
axis equal; axis tight;

figure;
plot(iris(index1, index(3)), iris(index1, index(4)), '*', ...
     iris(index2, index(3)), iris(index2, index(4)), 'o', ...
     iris(index3, index(3)), iris(index3, index(4)), 'x');
legend('Class 1', 'Class 2', 'Class 3');
title(sprintf('IRIS data onto the %s and %s feature (Laplacian Score)', num2str(index(3)), num2str(index(4))));
axis equal; axis tight;

disp('Laplacian Score:');
for i = 1:length(Score)
    disp(num2str(Score(i)));
end