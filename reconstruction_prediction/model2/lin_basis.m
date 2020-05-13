function [omega_lin]= lin_basis(x,y)% 
n_x=length(x);
n_y=length(y);
for i=1:n_x
    for j=1:n_y
            omega_lin(i,j)=min(x(i),y(j));
    end
end
