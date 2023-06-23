function label = Y2label(Y)
normat = repmat(sqrt(sum(Y.*Y,2)),1,size(Y,2));
for i = 1:size(normat,1)
    if (normat(i,1)==0)
        normat(i,:) = 1;
    end
end
label = normat;
end