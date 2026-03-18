d5(o3).
d5(o1).
d3(o5).
d4(o3).
d3(o0).

d4(V1) :- d3(V1).
d1(V1) :- d5(V1); not d2(V1).
d2(V1) :- d4(V1); not d3(V1).