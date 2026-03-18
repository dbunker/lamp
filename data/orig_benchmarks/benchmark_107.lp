d5(o0).
d4(o2).
d2(o5).
d5(o5).
d1(o1).
d3(o4).

d2(X) :- not d0(X), d3(X).
d2(X) :- d4(X), not d3(X).
d0(X) :- d4(X), not d1(X).
d4(X) :- d2(X), not d3(X).
d3(X) :- d4(X), not d5(X).
d2(X) :- not d1(X), d3(X).
d5(X) :- d2(X), not d3(X).
d5(X) :- d0(X), not d3(X).
d5(X) :- d2(X), not d1(X).
d4(X) :- not d5(X), d3(X).
d0(X) :- d2(X), not d1(X).