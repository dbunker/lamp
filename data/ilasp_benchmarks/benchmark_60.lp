d1(o0).
d5(o5).
d4(o5).
d3(o3).
d5(o4).
d3(o0).
d4(o1).
d3(o5).
d5(o3).

d3(V1) :- d5(V1).
d1(V1) :- d5(V1); not d4(V1).
d2(V1) :- d4(V1); not d5(V1).