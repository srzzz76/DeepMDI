function out = unwrap(wphs, tol, iter)
% UNWRAP  Robust phase unwrapping using a weighted least-squares method.
%
%   This implementation follows the algorithm described in:
%   Guo, Y., Chen, X., & Zhang, T. 
%   "Robust phase unwrapping algorithm based on least squares,"
%   Optics and Lasers in Engineering, 63, 25–29 (2014).
%
%   INPUT:
%       wphs.data : Wrapped phase map (in radians)
%       wphs.mask : Binary mask indicating valid pixels (1 = inside pupil)
%       tol       : Tolerance for convergence (e.g., 1e-6)
%       iter      : Maximum number of iterations (e.g., 100)
%
%   OUTPUT:
%       out.data  : Unwrapped phase map
%       out.mask  : Mask of valid pixels
%       out.type  : 'phase'
%
%   Example:
%       out = unwrap(wphs, 1e-6, 100);

% ------------------ Argument Handling ------------------
switch nargin
    case 1
        tol = 1e-3;
        iter = 10;
    case 2
        iter = 10;
end

% ------------------ Extract Valid Region ------------------
[m, n] = find(wphs.mask);
minr = min(m); maxr = max(m);
minc = min(n); maxc = max(n);

data = wphs.data(minr:maxr, minc:maxc);
mask = wphs.mask(minr:maxr, minc:maxc);
[rows, cols] = size(data);

% ------------------ Compute Wrapped Phase Gradients ------------------
% Horizontal gradient (x-direction)
dx1 = [diff(data,1,2), zeros(rows,1)];
dx1 = mod(dx1 + pi, 2*pi) - pi;
dx2 = circshift(dx1, [0 1]);

% Vertical gradient (y-direction)
dy1 = [diff(data); zeros(1,cols)];
dy1 = mod(dy1 + pi, 2*pi) - pi;
dy2 = circshift(dy1, [1 0]);

% ------------------ Compute Weights ------------------
% These weights ensure gradients are computed only within valid mask areas
wx1 = double([mask(:,1:end-1) & mask(:,2:end), mask(:,end)]);
wx2 = double([mask(:,1), mask(:,1:end-1) & mask(:,2:end)]);
wy1 = double([mask(1:end-1,:) & mask(2:end,:); mask(end,:)]);
wy2 = double([mask(1,:); mask(1:end-1,:) & mask(2:end,:)]);

% ------------------ Compute Residual (Laplacian) ------------------
r = (wx1.*dx1 - wx2.*dx2) + (wy1.*dy1 - wy2.*dy2);

% Construct the discrete Laplacian operator in the frequency domain
t = 2*( repmat(cos((0:rows-1)'*(pi/rows)), 1, cols) + ...
        repmat(cos((0:cols-1) *(pi/cols)), rows, 1) - 2 );
t(1,1) = 1;

data(:) = 0;
crit = tol * norm(r, 'fro');   % Convergence criterion

% ------------------ Iterative Conjugate Gradient Solver ------------------
for k = 1:iter
    % Solve Laplacian equation using DCT (fast Poisson solver)
    z = idct2(dct2(r) ./ t);
    b1 = sum(sum(r .* z));
    if k == 1
        p = z;
    else
        p = z + (b1 / b0) * p;
    end
    b0 = b1;

    % Compute gradient of p
    dx1 = [diff(p,1,2), zeros(rows,1)];
    dx2 = circshift(dx1,[0 1]);
    dy1 = [diff(p); zeros(1,cols)];
    dy2 = circshift(dy1,[1 0]);

    % Compute residual for current iteration
    rp = (wx1.*dx1 - wx2.*dx2) + (wy1.*dy1 - wy2.*dy2);
    a = sum(sum(p .* rp));
    r = r - (b1/a) * rp;
    data = data + (b1/a) * p;

    % Check convergence
    if norm(r, 'fro') < crit
        break;
    end
end

% ------------------ Assemble Output ------------------
out = wphs;
out.data(minr:maxr, minc:maxc) = data .* mask;
out.mask(minr:maxr, minc:maxc) = mask;
out.type = 'phase';

end
