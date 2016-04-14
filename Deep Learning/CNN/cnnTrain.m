function [tempcnn] = cnnTrain(X, Y, cnn, sc_connection, alpha)
        tempcnn = cnn;
    for i = 1:size(X,1)
        tic
        %cnn = forwardProp(reshape(X(i,:), cnn.layer{1,3}(1), cnn.layer{1,3}(2)), cnn, sc_connection); % for face
        tempcnn = forwardProp(reshape(X(i,:), tempcnn.layer{1,3}(1), tempcnn.layer{1,3}(2))', tempcnn, sc_connection); 
        
%         disp(tempcnn.layer{6,2});
%         disp(tempcnn.weight{2,1}{1});
        %%%%
        pred = tanh(cell2mat(tempcnn.layer{6,2})/2);
%         disp(cnn.weight{10,1}{1}(1,1:7) )        
%         disp(cnn.weight{9,1})
%         disp(cnn.weight{8,1}{1}(1,1:3) )
%         disp(cnn.weight{7,1})
%         disp(tempcnn.weight{6,1}{1}(1,1:3) )
%         disp(tempcnn.weight{5,1})
%         disp(tempcnn.weight{4,1}{1}(1,1:5) )
%         disp(tempcnn.weight{3,1})
%         disp(tempcnn.weight{2,1}{1}(1,1:7) )
%         
%         
        
        
        
        
        
        
        %pred(pred > 0) = 1;
        %pred(pred <= 0) = -1;

        y = Y(i,:)';
        error = mean((pred - y).^2);        
        disp(sprintf('Image %3.0d SSE %3.5f ',i,error));
        
         disp([pred y]);
%         
% 
%         toc
        %disp(y);
        figure(1)
        colormap gray;
        imagesc(tempcnn.layer{1,2}{1})
        
        figure(2)
        colormap gray;
        for p = 1:25
            subplot(5,5,p)
            imagesc(tempcnn.layer{3,2}{p})
        end

        figure(3)
        colormap gray;
        for p = 1:16
            subplot(4,4,p)
            imagesc(tempcnn.layer{5,2}{p})
        end
% 
%         figure(4)
%         colormap gray;
%         for p = 1:25
%             subplot(5,5,p)
%             imagesc(cnn.layer{7,2}{p})
%         end
        figure(6)
        colormap gray;
        for p = 1:25
            subplot(5,5,p)
            imagesc(tempcnn.weight{2,p}{:})
        end
        
        figure(7)
        colormap gray;
        for p = 1:25
            subplot(5,5,p)
            imagesc(tempcnn.weight{4,1}{p})
        end
%         
%         figure(5)
%         colormap gray;
%         counter = 0;
%         for p = 1:9
%             for pp = 1:length(tempcnn.weight{4,p})
%                 counter = counter +1;
%                 subplot(25,9,counter);
%                 imagesc(tempcnn.weight{4,p}{pp});
%             end
%         end
        %

%         
%         figure(5)
%         colormap gray;
%         for p = 1:25
%             subplot(5,5,p)
%             imagesc(cnn.layer{9,2}{p})
%         end

         drawnow;
        
        %disp(cnn.layer{6,2}{1});
        tempcnn = backwardProp(Y(i,:), tempcnn, sc_connection, alpha);

    end
end

