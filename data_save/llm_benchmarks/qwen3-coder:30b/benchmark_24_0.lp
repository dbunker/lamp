d2(o0).
d1(o0).
d0(o2).
d1(o2).

d0(X) :- d0(X), not d1(X).
d1(X) :- d2(X).
d0(X) :- d1(X).