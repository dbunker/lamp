d9(o0).
d0(o5).
d5(o1).
d0(o1).
d3(o4).
d7(o3).

d2(V) :- d0(V), d5(V).
d7(V) :- d0(V), not d5(V).
d8(V) :- not d0(V), d7(V).
d6(V) :- not d7(V), d5(V).