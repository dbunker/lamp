d1(o0).
d2(o1).
d1(o2).
d0(o0).

d0(V1) :- d2(V1).
d3(V1) :- d1(V1); not d0(V1).