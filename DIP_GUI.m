function DIP_GUI_()
    f = figure('Name', 'Deep Image Prior - Multi Task GUI', 'Position', [50 50 1250 750]);

    % UI Controls
    uicontrol('Style', 'text', 'Position', [450 690 200 30], 'String', 'Select an Image');
    uicontrol('Style', 'pushbutton', 'String', 'Browse Image', ...
        'Position', [470 660 150 30], 'Callback', @loadImage);

    uicontrol('Style', 'text', 'Position', [470 620 150 20], 'String', 'Select Task');
    taskMenu = uicontrol('Style', 'popupmenu', ...
        'String', {'Denoising','Deblurring','Super-Resolution','Inpainting'}, ...
        'Position', [470 590 150 25], 'Callback', @toggleSliders);

    uicontrol('Style', 'text', 'Position', [650 620 150 20], 'String', 'Image Size');
    sizeMenu = uicontrol('Style', 'popupmenu', ...
        'String', {'64', '128', '256'}, 'Value', 2, ...
        'Position', [650 590 100 25]);

    % Noise and Blur sliders
    noiseSlider = uicontrol('Style', 'slider', 'Position', [800 630 200 20], ...
        'Min', 0.01, 'Max', 0.2, 'Value', 0.1, 'Visible', 'on');
    noiseLabel = uicontrol('Style', 'text', 'Position', [800 655 200 20], ...
        'String', 'Noise Level: 0.10', 'Visible', 'on');

    blurSlider = uicontrol('Style', 'slider', 'Position', [800 570 200 20], ...
        'Min', 0.5, 'Max', 3.0, 'Value', 1.5, 'Visible', 'off');
    blurLabel = uicontrol('Style', 'text', 'Position', [800 595 200 20], ...
        'String', 'Blur Sigma: 1.5', 'Visible', 'off');

    addlistener(noiseSlider, 'Value', 'PostSet', @(~,~) updateSliderText(noiseSlider, noiseLabel, 'Noise Level: %.2f'));
    addlistener(blurSlider, 'Value', 'PostSet', @(~,~) updateSliderText(blurSlider, blurLabel, 'Blur Sigma: %.1f'));

    uicontrol('Style', 'pushbutton', 'String', 'Run DIP', ...
        'Position', [470 550 150 30], 'Callback', @runDIP);

    uicontrol('Style', 'pushbutton', 'String', 'Save Output Image', ...
        'Position', [470 510 150 30], 'Callback', @saveOutput);

    uicontrol('Style', 'pushbutton', 'String', 'Export Loss Curve', ...
        'Position', [470 470 150 30], 'Callback', @exportLossPlot);

    % Axes for image display
    ax1 = axes('Parent', f, 'Units', 'pixels', 'Position', [50 200 400 400]);
    title(ax1, 'Corrupted Image');

    ax2 = axes('Parent', f, 'Units', 'pixels', 'Position', [650 200 400 400]);
    title(ax2, 'DIP Output');

    ax3 = axes('Parent', f, 'Units', 'pixels', 'Position', [1100 280 250 200]);
    title(ax3, 'Loss Curve');

    psnrText = uicontrol('Style', 'text', 'Position', [930 230 250 25], 'String', 'PSNR: ');
    ssimText = uicontrol('Style', 'text', 'Position', [930 200 250 25], 'String', 'SSIM: ');

    % Initialize variables
    img = []; corrupted = []; mask = []; outputImg = []; losses = [];

    function loadImage(~, ~)
        [file, path] = uigetfile({'*.jpg;*.jpeg;*.png;*.tif'}, 'Select Image');
        if isequal(file, 0), return; end
        img = im2double(imread(fullfile(path, file)));
        sizeOpt = str2double(sizeMenu.String{sizeMenu.Value});
        img = imresize(img, [sizeOpt sizeOpt]);
        if size(img, 3) == 1, img = repmat(img, 1, 1, 3); end
        axes(ax1); imshow(img); title('Original Image'); axis image;
    end

    function toggleSliders(~, ~)
        task = taskMenu.Value;
        set(noiseSlider, 'Visible', strcmp(task, '1'));
        set(noiseLabel, 'Visible', strcmp(task, '1'));
        set(blurSlider,  'Visible', strcmp(task, '2'));
        set(blurLabel,  'Visible', strcmp(task, '2'));
    end

    function runDIP(~, ~)
        if isempty(img), return; end
        sizeOpt = size(img, 1);  % 64, 128, 256
        task = taskMenu.Value;
        mask = [];

        switch task
            case 1  % Denoising
                noiseLevel = noiseSlider.Value;
                corrupted = img + noiseLevel * randn(size(img));
                corrupted = max(min(corrupted, 1), 0);
                target = corrupted;
                lossFunc = @(Y,T) mse(Y, T);
            case 2  % Deblurring
                sigma = blurSlider.Value;
                psf = fspecial('gaussian', 5, sigma);
                corrupted = imfilter(img, psf, 'conv', 'replicate');
                target = corrupted;
                lossFunc = @(Y,T) mse(Y, T);
            case 3  % Super-resolution
                low_res = imresize(img, 0.5, 'bicubic');
                corrupted = imresize(low_res, [sizeOpt sizeOpt], 'bicubic');
                target = corrupted;
                lossFunc = @(Y,~) mse(imresize(Y, 0.5, 'bicubic'), low_res);
            case 4  % Inpainting
                mask = rand(size(img,1), size(img,2)) > 0.5;
                mask = repmat(mask, [1 1 3]);
                corrupted = img .* mask;
                target = img;
                lossFunc = @(Y,T) mse(Y .* mask, T .* mask);
        end

        axes(ax1); imshow(corrupted); title('Corrupted Image'); axis image;

        % Initialize DIP
        z = rand([sizeOpt sizeOpt 3]);
        dlZ = dlarray(single(z), 'SSC');
        dlTarget = dlarray(single(target), 'SSC');

        % U-Net & dlnetwork (2025 fix)
        dagNet = unet([sizeOpt sizeOpt 3], 3);
        lgraph = layerGraph(dagNet);
        idx = find(arrayfun(@(l) isa(l, 'nnet.cnn.layer.PixelClassificationLayer') || ...
                                 isa(l, 'nnet.cnn.layer.ClassificationOutputLayer'), ...
                            lgraph.Layers));
        if ~isempty(idx)
            lgraph = removeLayers(lgraph, lgraph.Layers(idx).Name);
        end
        net = dlnetwork(lgraph);

        lr = 1e-2; trailingAvg = []; trailingAvgSq = [];
        losses = [];

        h = waitbar(0, 'Running DIP Optimization...');
        for i = 1:300
            [loss, grads] = dlfeval(@modelGradients, net, dlZ, dlTarget, lossFunc);
            [net, trailingAvg, trailingAvgSq] = adamupdate(net, grads, ...
                trailingAvg, trailingAvgSq, i, lr);
            losses(end+1) = double(gather(extractdata(loss)));
            waitbar(i/300, h, sprintf('Iteration %d / 300, Loss: %.4f', i, losses(end)));

            if mod(i, 10) == 0
                Y = extractdata(predict(net, dlZ));
                Y = max(min(Y, 1), 0);  % Clamp output
                axes(ax2); imshow(Y); title(sprintf('DIP Output (Iter %d)', i)); axis image;

                axes(ax3); cla;
                plot(losses, 'LineWidth', 1.5); grid on;
                title('Loss over Iterations');
                xlabel('Iteration'); ylabel('Loss');
                drawnow;
            end
        end
        close(h);

        % Final Output
        Y = extractdata(predict(net, dlZ));
        Y = max(min(Y, 1), 0);
        outputImg = Y;
        axes(ax2); imshow(Y); title('Final DIP Output'); axis image;

        % Metrics
        if task ~= 3
            psnr_val = psnr(Y, img);
            ssim_val = ssim(Y, img);
        else
            ref = imresize(imresize(img, 0.5, 'bicubic'), [sizeOpt sizeOpt], 'bicubic');
            psnr_val = psnr(Y, ref);
            ssim_val = ssim(Y, ref);
        end
        psnrText.String = sprintf('PSNR: %.2f dB', psnr_val);
        ssimText.String = sprintf('SSIM: %.3f', ssim_val);
    end

    function saveOutput(~, ~)
        if isempty(outputImg), return; end
        [file, path] = uiputfile({'*.png'; '*.jpg'}, 'Save Output As');
        if isequal(file, 0), return; end
        imwrite(outputImg, fullfile(path, file));
    end

    function exportLossPlot(~, ~)
        if isempty(losses), return; end
        [file, path] = uiputfile('loss_curve.png', 'Save Loss Curve As');
        if isequal(file, 0), return; end
        fig = figure('Visible', 'off');
        plot(losses, 'LineWidth', 1.5); grid on;
        title('Loss over Iterations'); xlabel('Iteration'); ylabel('Loss');
        saveas(fig, fullfile(path, file));
        close(fig);
    end

    function loss = mse(Y, T)
        loss = mean((Y - T).^2, 'all');
    end

    function [loss, grads] = modelGradients(net, dlZ, target, lossFunc)
        Y = forward(net, dlZ);
        loss = lossFunc(Y, target);
        grads = dlgradient(loss, net.Learnables);
    end

    function updateSliderText(slider, label, fmt)
        val = get(slider, 'Value');
        set(label, 'String', sprintf(fmt, val));
    end
end
