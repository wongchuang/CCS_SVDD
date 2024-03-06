function Q=Update_Q(Q,Traindata,Alphavector,beta,delta,miu1,miu2,numofmodes,eta,GPU)

    M_num = cell(1,numofmodes);      
    W_intra_M = cell(1,numofmodes);
    D_intra_M = cell(1,numofmodes);
    temp_M = cell(1,numofmodes);
    index_M = cell(1,numofmodes);
    L_intra_M = cell(1,numofmodes);
    W_inter = ones(size(Traindata{1},2),size(Traindata{1},2));      
    D_inter = zeros(size(Traindata{1},2), size(Traindata{1},2));
    for i = 1:size(Traindata{1},2)
        D_inter(i, i) = sum(W_inter(i, :));  
    end
    L_inter = D_inter - W_inter;    
    for i = 1:numofmodes      
        M_num{i} = size(Traindata{i},2);
        W_intra_M{i} = zeros(M_num{i}, M_num{i});
        D_intra_M{i} = zeros(M_num{i}, M_num{i});
        temp_M{i} = zeros(M_num{i}, M_num{i});
        for ii = 1:M_num{i}   
            for jj = 1:M_num{i}
                temp_M{i}(ii, jj) = sum((Traindata{i}(:, ii) - Traindata{i}(:, jj)).^2);
            end
        end
        [~, index_M{i}] = sort(temp_M{i}, 2);    
        k_num = M_num{i};
        for ii = 1:M_num{i}                           
            for jj = 1:k_num                
                W_intra_M{i}(ii, index_M{i}(ii, jj)) = exp(-temp_M{i}(ii, index_M{i}(ii, jj)) ./ (2 .* (delta .^ 2)));
            end
        end
        for ii = 1:M_num{i}  
            for jj = 1:M_num{i}
                if W_intra_M{i}(ii, jj) > 0
                    D_intra_M{i}(ii, ii) = D_intra_M{i}(ii, ii) + 1;
                end
            end
        end
        L_intra_M{i} = D_intra_M{i} - beta .* W_intra_M{i};         
    end
    for m = 1:numofmodes
        R = zeros(size(Traindata{m},1), size(Traindata{m},1));
        QXLX = zeros(size(Q{m},1),size(Q{m},2));
        aaQxx = zeros(size(Q{m},1),size(Q{m},2));
        aQxx = zeros(size(Q{m},1),size(Q{m},2));
        small_value = 0.00001;
        temp = sum(Q{m}.^2, 1)';
        for num = 1:size(Traindata{m},1)
            R(num, num) = 1./(2 .* sqrt(temp(num, 1) + small_value));
        end
        QR = Q{m} * R;       
        for p=1:numofmodes       
            if p~=m
                QXLX = QXLX + Q{p} * Traindata{p} * L_inter * (Traindata{m}');
            else
                QXLX = QXLX + Q{m} * Traindata{m} * L_intra_M{m} * (Traindata{m}');
            end
        end
        if GPU                                
            aaQxx = gpuArray(aaQxx);
            aQxx = gpuArray(aQxx);
            Q{m} = gpuArray(Q{m});
            for p = 1:numofmodes
                Alphavector{p} = gpuArray(Alphavector{p});
                Traindata{p} = gpuArray(Traindata{p});
            end
        end
        for p = 1:numofmodes
            for j = 1:size(Alphavector{p},1)   
                for i = 1:size(Alphavector{m},1)   
                    if (Alphavector{m}(i)>0) && (Alphavector{p}(j)>0)    
                        aaQxx=aaQxx +  Alphavector{m}(i) .* Alphavector{p}(j) .* Q{p} * Traindata{p}(:,j) * (Traindata{m}(:,i)');
                    end
                end
            end
        end
        for i = 1:size(Alphavector{m},1) 
            if Alphavector{m}(i)>0       
                aQxx=aQxx +  Alphavector{m}(i) .* Q{m} * Traindata{m}(:,i) * (Traindata{m}(:,i)');
            end
        end
        if GPU                                  
            aaQxx = gather(aaQxx);
            aQxx = gather(aQxx);
            Q{m} = gather(Q{m});
            for p = 1:numofmodes
                Alphavector{p} = gather(Alphavector{p});
                Traindata{p} = gather(Traindata{p});
            end
        end
        Q{m}=Q{m} - 2*eta .* (miu1 * QXLX + miu2 * QR + aQxx - aaQxx);
    end
end
