function[omega]=thinplate_basis(t1,s1,t2,s2)
n1=length(t1);
n2=length(t2);
for  j=1:n2
	     for i=1:n1
	        rij=sqrt((t1(i)-t2(j))^2+(s1(i)-s2(j))^2);
	        p1i=1.0-2.0*t1(i)-2.0*s1(i);
	        p1j=1.0-2.0*t2(j)-2.0*s2(j);
	        p2i=2.0*t1(i);
	        p2j=2.0*t2(j);
	        p3i=2.0*s1(i);
	        p3j=2.0*s2(j);	
	        rs1i=sqrt((0.0-t1(i))^2+(0.0-s1(i))^2);
	        rs1j=sqrt((0.0-t2(j))^2+(0.0-s2(j))^2);
	        rs2i=sqrt((0.50-t1(i))^2+(0.0-s1(i))^2);
	        rs2j=sqrt((0.50-t2(j))^2+(0.0-s2(j))^2);
	        rs3i=sqrt((0.0-t1(i))^2+(0.50-s1(i))^2);
	        rs3j=sqrt((0.0-t2(j))^2+(0.50-s2(j))^2);
	      	rs1s2=sqrt((0.0-0.50)^2+(0.0-0.0)^2);
	      	rs2s3=sqrt((0.50-0.0)^2+(0.0-0.50)^2);
	      	rs1s3=sqrt((0.0-0.0)^2+(0.0-0.50)^2);
	
	      	if(rij>0.0000001) 
	          Aij=rij*log(rij);
	      	else
	          Aij=0.0;
	      	end

	      	if(rs1i>0.0000001) 
                  As1i=rs1i*log(rs1i);
              	else
                  As1i=0.0;
              	end
	      
	      	if(rs1j>0.0000001) 
                  As1j=rs1j*log(rs1j);
              	else
                  As1j=0.0;
              	end

	      	if(rs2i>0.0000001) 
                  As2i=rs2i*log(rs2i);
              	else
                  As2i=0.0;
              	end

	      	if(rs2j>0.0000001) 
                  As2j=rs2j*log(rs2j);
              	else
                  As2j=0.0;
              	end

	      	if(rs3i>0.0000001) 
                  As3i=rs3i*log(rs3i);
              	else
                  As3i=0.0;
              	end

	      	if(rs3j>0.0000001) 
                  As3j=rs3j*log(rs3j);
              	else
                  As3j=0.0;
              	end

	      	if(rs1s2>0.0000001) 
                  As1s2=rs1s2*log(rs1s2);
              	else
                  As1s2=0.0;
              	end

	      	if(rs2s3>0.0000001) 
                  As2s3=rs2s3*log(rs2s3);
              	else
                  As2s3=0.0;
              	end

	      	if(rs1s3>0.0000001) 
                  As1s3=rs1s3*log(rs1s3);
              	else
                  As1s3=0.0;
               	end

	        omega(i,j)=Aij-p1j*As1i-p2j*As2i-p3j*As3i-...
     	        p1i*As1j-p2i*As2j-p3i*As3j+p1i*p2j*As1s2+...
     	        p1i*p3j*As1s3+p2i*p1j*As1s2+p2i*p3j*As2s3+...
     		p3i*p1j*As1s3+p3i*p2j*As2s3;
	  end
 end