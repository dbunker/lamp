d1(o4).
d0(o4).
d6(o4).
d6(o5).
d8(o4).
d5(o3).

d9(V) :- d0(V).
d7(V) :- d4(V).
d4(V) :- d1(V).
d3(V) :- d0(V), d6(V).
d2(V) :- d0(V).
d5(V) :- d5(U), d1(V).