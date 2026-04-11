d5(o0).
d5(o1).
d2(o4).
d2(o0).
d1(o2).

d0(X) :- d2(X).
d1(X) :- d5(X), not d2(X).
d5(X) :- d1(X).