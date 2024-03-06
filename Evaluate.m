function EVAL=Evaluate(Q,Testdata,Testlabel,svdd,numofmodes)
    predict_label = cell(1,numofmodes);
    for i = 1:numofmodes
        RedTestdata=Q{i} * Testdata{i};
        results = svdd.test(RedTestdata', Testlabel);
        predict_label{i} = results.predictedLabel;
    end
    EVAL = [];
    Decission_and = zeros(size(predict_label{1},1),1);
    for i = 1:numofmodes
        Decission_and = Decission_and + predict_label{i};
    end
    Decission_and(Decission_and==0) = -1;
    Decission_and(Decission_and==numofmodes) = 1;
    Decission_and(Decission_and==-numofmodes) = -1;
    mic_and= Evaluate_mic(Testlabel,Decission_and);
    EVAL = cat(2,EVAL,mic_and(1),mic_and(2),mic_and(3),mic_and(7));
    Decission_or = zeros(size(predict_label{1},1),1);
    for i = 1:numofmodes
        Decission_or = Decission_or + predict_label{i};
    end
    Decission_or(Decission_or~=-numofmodes) = 1;
    Decission_or(Decission_or==-numofmodes) = -1;
    mic_or= Evaluate_mic(Testlabel,Decission_or);
    EVAL = cat(2,EVAL,mic_or(1),mic_or(2),mic_or(3),mic_or(7));
    for i = 1:numofmodes
        mic= Evaluate_mic(Testlabel,predict_label{i});
        EVAL = cat(2,EVAL,mic(1),mic(2),mic(3),mic(7));
    end
end