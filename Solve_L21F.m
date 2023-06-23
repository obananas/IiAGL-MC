function [E] = Solve_L21F(W,gma)
% solve min{ gma||x||_21 + (1/2)||x-w||_F^2 }
n = size(W,2);
E = W;
for i=1:n
    E(:,i) = Solve_L2(W(:,i),gma);
end
end

function [x] = Solve_L2(w,gma)
% solve min{ gma||x||_2 + ||x-w||_2^2 }
nw = norm(w);
if nw > gma
    x = (nw - gma) * w / nw;
else
    x = zeros(length(w), 1);
end
end