function [D] = NB_XGivenY(XTrain, yTrain)
    D = zeros(length(unique(yTrain)), size(XTrain,2));
    
    for class = 1:size(D,1)
        %D(class,:) = (sum(XTrain(yTrain == class,:),1) + 1) ./ (sum(sum(XTrain(yTrain == class,:),1)) + 1);
        D(class,:) = (sum(XTrain(yTrain == class,:),1) + 1) ./ (sum(yTrain == class) + 1);
    end
end