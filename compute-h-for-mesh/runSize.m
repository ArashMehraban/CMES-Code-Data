clear
clc

meshes = {'../Beam8_0','../Beam8_1','../Beam8_2','../Beam8_3','../Beam8_15'};
hsz = zeros(1, size(meshes,2));

for i = 1:size(meshes,2)
    [~, msh] = get_mesh(meshes{i}, 'exo', 'lex');
    hsz(i) = get_hsz(msh);
end

disp(hsz)