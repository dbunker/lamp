d2(o3).
d1(o5).
d0(o5).
d0(o4).
d5(o1).
d0(o2).

d3(X) :- d4(X), not d1(X).
d4(X) :- d0(X).
d2(X) :- d5(X).
d5(X) :- d2(X).