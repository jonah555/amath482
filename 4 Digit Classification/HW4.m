%% Load data and reshape

[images, labels] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
[t_images, t_labels] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');

training = zeros(784,60000);
for i = 1:60000
    training(:,i) = im2double(reshape(images(:,:,i),784,1));
end

testing = zeros(784,10000);
for i = 1:10000
    testing(:,i) = im2double(reshape(t_images(:,:,i),784,1));
end


%% Singular Value Decomposition

[U,S,V] = svd(training ,'econ');

figure()
subplot(2,1,1)
plot(diag(S),'ko','Linewidth',2)
title("Singular Value Spectrum")
set(gca,'Fontsize',16)
subplot(2,1,2)
semilogy(diag(S),'ko','Linewidth',2)
title("Singular Value Spectrum (log scale)")
set(gca,'Fontsize',16)
saveas(gcf, 'spectrum.png')


%% Projection onto 3 V-modes

projections = U(:,[2,3,5])'*training;
for digit=0:9
    projection = projections(:,labels == digit); 
    plot3(projection(1,:),projection(2,:),projection(3,:),'o', ...
        'DisplayName', sprintf('%i',digit))
    hold on
end
title('Projection onto 3 V-modes')
xlabel('1st V-Mode') 
ylabel('2nd V-Mode')
zlabel('3rd V-Mode') 
legend
set(gca,'Fontsize',16)
saveas(gcf, 'projection.png')


%% Running LDA on all pairs

feature = 60;
accuracies=zeros(10,10);
correct = 0;
count = 0;
for i=0:8
    for j=i+1:9
        digit1 = training(:,labels==i);
        digit2 = training(:,labels==j);
        [U,~,~,threshold,w,~,~] = digits_trainer(digit1,digit2,feature);
        
        test1 = testing(:,t_labels==i);
        match=0;
        length1=size(test1,2);
        for k=1:length1
            digit = test1(:,k); 
            IMat = U' * digit; 
            digitval = w' * IMat;
            if digitval < threshold 
                match = match + 1;
            end
        end
        
        test2 = testing(:,t_labels==j);
        length2 = size(test2,2);
        for k=1:size(test2 ,2) 
            digit = test2(:,k); 
            IMat = U' * digit; 
            digitval = w' * IMat;
            if digitval > threshold 
                match = match + 1;
            end
        end
        
        accuracy = match/(length1+length2);
        accuracies(i+1,j+1) = accuracy;
        count = count+length1+length2; 
        correct = correct+match;
    end
end
LDA_accuracy = correct / count


%% Other Classifiers

[U,S,V] = svd(training,'econ'); 
U=U(:,1:60);
projection = S*V';
train = (U'*training)'./max(projection(:)); 
test = (U'*testing)'./max(projection(:)); 

% SVM (support vector machines)
Mdl = fitcecoc(train,labels);
result = predict(Mdl,test);
match = result == t_labels;
SVM_accuracy = sum(match)/size(match,1)

% Decision Tree Classifiers
d_tree = fitctree(train,labels);
result = predict(d_tree,test); 
match = result == t_labels; 
tree_accuracy = sum(match)/size(match,1)


%% SVM (support vector machines)

train_data=train';
test_data=test';

train_0 = train_data(:,labels==0);
train_4 = train_data(:,labels==4);
train_9 = train_data(:,labels==9);

test_0 = test_data(:,t_labels==0); 
test_4 = test_data(:,t_labels==4); 
test_9 = test_data(:,t_labels==9);

label_0 = zeros(1,size(train_0,2));
label_4 = zeros(1,size(train_4,2)) + 4;
label_9 = zeros(1,size(train_9,2)) + 9;

t_label_0 = zeros(1,size(test_0 ,2));
t_label_4 = zeros(1,size(test_4 ,2)) + 4; 
t_label_9 = zeros(1,size(test_9 ,2)) + 9;


train_04 = [train_0 train_4]; 
test_04 = [test_0 test_4]; 
label_04 = [label_0 label_4]; 
t_label_04 = [t_label_0 t_label_4];

Mdl_04 = fitcsvm(train_04',label_04);
results = predict(Mdl_04,test_04'); 
match = results == t_label_04'; 
SVM_easy_accuracy = sum(match)/size(match,1)


train_49 = [train_4 train_9]; 
test_49 = [test_4 test_9]; 
label_49 = [label_4 label_9];
t_label_49 = [t_label_4 t_label_9];

Mdl_49 = fitcsvm(train_49',label_49);
results = predict(Mdl_49,test_49'); 
match = results == t_label_49'; 
SVM_hard_accuracy =sum(match)/size(match,1)


%% Decision Tree Classifiers

d_tree1 = fitctree(train_04',label_04);
results = predict(d_tree1 ,test_04'); 
match = results == t_label_04'; 
tree_easy_accuracy = sum(match)/size(match,1)

d_tree2 = fitctree(train_49', label_49); 
results = predict(d_tree2 ,test_04');
match = results == t_label_04';
tree_hard_accuracy = sum(match)/size(match,1)


%% digits_trainer function

function [U,S,V,threshold,w,sort1,sort2] = digits_trainer(d1,d2,feature)
    n1 = size(d1,2) ;
    n2 = size(d2,2) ;
    [U,S,V] = svd([d1 d2],'econ');
    digits = S*V';
    U = U(:,1:feature);
    digit1 = digits(1:feature,1:n1);
    digit2 = digits(1:feature,n1+1:n1+n2);
    ma = mean(digit1,2);
    mb = mean(digit2,2);
    
    Sw = 0;
    for k=1:n1
        Sw = Sw + (digit1(:,k)-ma)*(digit1(:,k)-ma)';
    end
    for k=1:n2
        Sw = Sw + (digit2(:,k)-mb)*(digit2(:,k)-mb)';
    end
    Sb = (ma-mb)*(ma-mb)';
    
    [V2,D] = eig(Sb,Sw);
    [~,ind] = max(abs(diag(D)));
    w = V2(:,ind);
    w = w / norm(w,2) ;
    v1 = w'*digit1;
    v2 = w'*digit2;
    
    if mean(v1) > mean(v2)
        w = -w;
        v1 = -v1;
        v2 = -v2;
    end
    
    sort1 = sort(v1);
    sort2 = sort(v2);
    t1 = length(sort1);
    t2 = 1;
    
    while sort1(t1) > sort2(t2)
        t1 = t1-1;
        t2 = t2+1;
    end
    threshold = (sort1(t1) + sort2(t2))/2;
end