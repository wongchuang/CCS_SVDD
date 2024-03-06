function PlotData(Q,Traindata,R,center,Testdata)
    subplot(1, 2, 1);
    aaaa=Traindata{1}';
    bbbb=Traindata{2}';
    cccc=Testdata{1}';
    dddd=Testdata{2}';
    plot(aaaa(:,1),aaaa(:,2),'bo','MarkerFaceColor','b');
    plot(bbbb(:,1),bbbb(:,2),'bo','MarkerFaceColor','b');
    hold on;
    plot(cccc(:,1),cccc(:,2),'bo','MarkerFaceColor','g');
    plot(dddd(:,1),dddd(:,2),'bo','MarkerFaceColor','g');
  
    subplot(1, 2, 2)
    a{1} = (Q{1} * Traindata{1})';
    a{2} = (Q{2} * Traindata{2})';
    b{1} = (Q{1} * Testdata{1})';
    b{2} = (Q{2} * Testdata{2})';
    plot(a{1}(:,1),a{1}(:,2),'bo','MarkerFaceColor','b');
    plot(a{2}(:,1),a{2}(:,2),'bo','MarkerFaceColor','b');
    hold on;
    plot(b{1}(:,1),b{1}(:,2),'bo','MarkerFaceColor','g');
    plot(b{2}(:,1),b{2}(:,2),'bo','MarkerFaceColor','g');
    hold on;
    theta=0:0.01:2*pi;
    x=center(1)+R*cos(theta);
    y=center(2)+R*sin(theta);
    plot(x,y,'k','linewidth',1)
    title('二维圆形边界—无负样本无核函数')
    axis equal
end
