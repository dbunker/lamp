d3(o1).
d3(o2).
d4(o2).
d0(o1).
d0(o2).
d4(o5).
d3(o3).
d1(o2).
d0(o3).

d5(V1) :- d3(V1); not d4(V1).
d2(V1) :- d4(V1); not d3(V1).