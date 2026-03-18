d2(o2).
d3(o2).
d4(o2).
d5(o1).
d3(o5).
d3(o4).
d2(o0).
d2(o1).
d0(o4).

d4(V1) :- d0(V1).
d1(V1) :- d2(V1); not d4(V1).
d0(V1) :- d3(V1); not d2(V1).
d3(V1) :- d1(V1); not d5(V1).