d4(o4).
d0(o1).
d0(o2).
d2(o1).

d4(X) :- d0(X).
d2(X) :- d4(X).
d0(X) :- d2(X).
d3(X) :- d0(X), not d2(X).