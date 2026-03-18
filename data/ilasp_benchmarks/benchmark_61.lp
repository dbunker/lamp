d1(o4).
d0(o5).
d4(o5).
d5(o4).
d2(o4).
d2(o1).
d3(o5).
d4(o2).
d4(o3).

d3(V1) :- d5(V1).
d2(V1) :- d0(V1).
d1(V1) :- d2(V1); not d0(V1).
d4(V1) :- d2(V1); not d5(V1).