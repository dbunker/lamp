d4(o0).
d2(o0).
d1(o2).
d3(o1).

d4(X) :- not d3(X), not d1(X), not d0(X), d2(X).
d2(X) :- d4(X), not d3(X), not d1(X), not d0(X).
d1(X) :- not d4(X), not d3(X), not d0(X), d2(X).
d3(X) :- not d2(X), not d0(X), d1(X), not d4(X).
d0(X) :- d3(X), not d1(X), not d2(X).