function  [yHat] = NB_Classify(D, p, X)
    
    prob = zeros(size(X,1),size(X,2),2);
    
    for example = 1:size(prob,1)
        for class = 1:size(prob,3)
            prob(example,:,class) = D(class,:) .* X(example,:) + ((1 - D(class,:)) .* (1 - X(example,:)));
        end
    end
    
    log_probs = sum(log(prob),2);
    log_post = [(log_probs(:,1) + log(p)), (log_probs(:,2) + log(1-p))];
    
    prob_diff = log_post(:,1) - log_post(:,2);
    inds = prob_diff > 0;
    yHat(inds) = 1;
    yHat(~inds) = 2;
    
end