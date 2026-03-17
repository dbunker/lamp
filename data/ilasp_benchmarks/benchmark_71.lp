d0(o3).
d3(o4).
d1(o0).
d2(o2).
d0(o4).

d0(V1) :- d2(V1).
d2(V1) :- d0(V1); not d3(V1).