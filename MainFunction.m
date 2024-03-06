function [Q,svdd]=MainFunction(Traindata,Trainlabel,d,numofmodes,maxIter,C,beta,delta,miu1,miu2,eta,GPU)
    for i=1:numofmodes    
        tempQ = pca(Traindata{i}');
        Q{i}=tempQ(:,1:d)';
        y{i} = Q{i} * Traindata{i};
    end
    
    for iter = 1:maxIter
        y_data = [];y_label=[];
        for i=1:numofmodes  
            y_data = cat(2,y_data,y{i});
            y_label = cat(1,y_label,Trainlabel);
        end
        
        kernel = Kernel_SVDD('type', 'linear');
        svddParameter = struct('cost', C,'kernelFunc', kernel,'display','off');
        svdd = BaseSVDD(svddParameter);
        svdd.train(y_data',y_label);
        BigAlphavector = svdd.alpha;
        jj=0;
        for i=1:numofmodes
            Alphavector{i}=BigAlphavector(jj+1:jj + size(y{i},2));
            jj = jj + size(y{i},2);
        end
        Q=Update_Q(Q,Traindata,Alphavector,beta,delta,miu1,miu2,numofmodes,eta,GPU);
        for i=1:numofmodes
            [tmpQ, ~]=qr(Q{i}',0);
            Q{i} = tmpQ';
            clear tmpQ R;
            tmpNorm = sqrt(diag(Q{i}*Q{i}'));
            Q{i} = Q{i}./(repmat(tmpNorm',size(Q{i},2),1)');
            y{i} = Q{i} * Traindata{i};
        end
     end
   y_data = [];y_label=[];
        for i=1:numofmodes  
            y_data = cat(2,y_data,y{i});
            y_label = cat(1,y_label,Trainlabel);
        end
    
    kernel = Kernel_SVDD('type', 'linear');
    svddParameter = struct('cost', C,'kernelFunc', kernel,'display','off');
    svdd = BaseSVDD(svddParameter);
    svdd.train(y_data',y_label);
end