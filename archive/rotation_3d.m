syms y p r

Ry = [cos(y) 0 sin(y); 0 1 0; -sin(y) 0 cos(y)] % yaw

Rx = [1 0 0; 0 cos(p) -sin(p); 0 sin(p) cos(p)] % pitch

Rz = [cos(r) -sin(r) 0; sin(r) cos(r) 0; 0 0 1] % roll

R = Ry*Rx*Rz

yaw = simplify(atan(R(1,3)/R(3,3)))
pitch = simplify(-atan(R(2,3)/sqrt(R(2,1).^2+R(2,2).^2)))
roll = simplify(atan(R(2,1)/R(2,2)))