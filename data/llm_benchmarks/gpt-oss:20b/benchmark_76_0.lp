d0(o0).
d1(o1).
d5(o1).
d2(o1).
d3(o5).
d4(o3).

d1(V) :- d3(V).
d0(V) :- not d2(V), d5(V).
d4(V) :- d2(V).
d5(V) :- d3(V).
d3(V) :- d0(V).