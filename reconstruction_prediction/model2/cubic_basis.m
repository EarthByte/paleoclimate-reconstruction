function [omega_cubic]= cubic_basis(x,y)% 
n_x=length(x);
n_y=length(y);
for i=1:n_x
    for j=1:n_y
        if(x(i) <= y(j)) 
            omega_cubic(i,j)=x(i)^2*(y(j)-x(i)/3)/2;
        else
            omega_cubic(i,j)=y(j)^2*(x(i)-y(j)/3)/2;
        end
    end
end
